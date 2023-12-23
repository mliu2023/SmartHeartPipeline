import torch
import torch.nn as nn
import numpy as np
import biosppy
from calculate_downsample import cnn_output_shape
from models.SEResNet import SEResNet

# Class for generating a feature vector from a heartbeat
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, activation):
        super(Encoder, self).__init__()
        layers = []
        bn_layers = []
        downsample_layers = []
        channels = in_channels 
        for i in range(len(kernel_sizes)):
            next_channels = channels
            if channels < out_channels:
                next_channels = min(next_channels*2, out_channels)
            layers.append(nn.Conv1d(channels, next_channels, kernel_sizes[i], strides[i], kernel_sizes[i]//2))
            bn_layers.append(nn.BatchNorm1d(next_channels))
            downsample_layers.append(nn.Conv1d(channels, next_channels, kernel_size=1, stride=strides[i]))
            channels = next_channels
        self.layers = nn.ModuleList(layers)
        self.bn_layers = nn.ModuleList(bn_layers)
        self.downsample_layers = nn.ModuleList(downsample_layers)
        self.activation = activation
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # num_time_steps = cnn_output_shape(self.layers, 450)
        #self.linear = nn.Linear(in_channels*300, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        #x = self.activation(self.linear(x))
        for i in range(len(self.layers)):
            identity = x
            x = self.layers[i](x)
            x = self.bn_layers[i](x)
            identity = self.downsample_layers[i](identity)
            x += identity
            x = self.activation(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        # x = x.view(x.size(0), -1, 1)
        return x

#Transformer based on feature vectors extracted from individual heartbeats
class ECGFormer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes, strides, activation, n_layers, attention_heads, num_additional_features, max_seq_length, n_classes, device):
        super(ECGFormer, self).__init__()
        self.encoder = Encoder(in_channels, out_channels, kernel_sizes, strides, activation)
        self.transformer_layer = nn.TransformerEncoderLayer(out_channels, attention_heads, activation='gelu')
        self.transformer = nn.TransformerEncoder(self.transformer_layer, n_layers)
        
        self.device = device
        self.max_seq_length = max_seq_length
        
        self.class_embeddings = torch.hstack([nn.Parameter((torch.normal(mean=0, std = 0.5, size=(out_channels, 1)).cuda())) for _ in range(n_classes)])
        # self.class_embeddings = torch.hstack([nn.Parameter(torch.normal(0, 1, (out_channels, 1).cuda())) for _ in range(n_classes)])
        result = torch.ones(max_seq_length+n_classes, out_channels)
        for i in range(max_seq_length):
            for j in range(out_channels):
                result[i][j] = np.sin(i / (10000 ** (j/out_channels))) if j % 2 == 0 else np.cos(i/(10000 ** ((j-1)/out_channels)))
        self.positional_embeddings = nn.Parameter(result)
        #embeddings for demographics
        self.demo_embed  = nn.Parameter(torch.normal(mean=0, std = 0.5, size=(27, num_additional_features)))
        self.dense = nn.Linear(out_channels+num_additional_features, 1)
        # self.decoder = nn.Sequential(
        #     nn.Conv1d(out_channels, out_channels, 5, 2),
        #     nn.ReLU(),
        #     nn.Conv1d(out_channels, out_channels, 5, 2),
        #     nn.ReLU(), 
        #     nn.Flatten(start_dim=1, end_dim=-1),
        #     nn.Linear(out_channels, 27)
        # )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x, demographics, rpeaks):
        x = x.cpu()
        signals = []
        for i in range(x.shape[0]):
            signal = []
            for j in range(12):
                before = 0.5
                after = 0
                template_length = int(before*500)+int(after*500)
                #print(i, j)
                #print(x.shape)
                before_template, _ = biosppy.signals.ecg.extract_heartbeats(signal=np.array(x[i][j]), rpeaks=rpeaks[i][j], sampling_rate=500, before=before, after=after)
                if len(before_template) == 0:
                    before_templates = x[i][j][0:template_length].unsqueeze(dim=0)
                else:
                    before_templates = torch.vstack([torch.tensor(before_template[i], dtype=torch.float) for i in range(min(len(before_template),self.max_seq_length))])
                before = 0
                after = 0.5
                template_length = int(before*500)+int(after*500)
                after_template, _ = biosppy.signals.ecg.extract_heartbeats(signal=np.array(x[i][j]), rpeaks=rpeaks[i][j], sampling_rate=500, before=before, after=after)
                if len(after_template) == 0:
                    templates = x[i][j][0:template_length].unsqueeze(dim=0)
                else:
                    templates = torch.vstack([torch.tensor(after_template[i], dtype=torch.float) for i in range(min(len(after_template),self.max_seq_length))])

                combined_signal = []
                for k in range(max(len(after_template), len(before_templates))):
                    if k < len(before_templates):
                        combined_signal.append(torch.tensor(before_templates[k]))
                    if k < len(after_template):
                        combined_signal.append(torch.tensor(after_template[k]))
                combined_signal = torch.stack(combined_signal)
                for k in range(len(combined_signal), self.max_seq_length):
                    combined_signal = torch.cat((combined_signal, torch.zeros(1, 500)))
                signal.append(combined_signal)

            ### use for debugging ####
            # import matplotlib.pyplot as plt
            # import os
            # plt.figure(figsize=(20, 15))
            # for i in range(0, 12):
            #     plt.subplot(3, 4, i+1)
            #     plt.plot(signal[i][3])
            # plt.savefig(os.path.join('data_visualization', 'template.png'))
            # exit(0)
            ###########################
            signal = torch.stack(signal)
            signals.append(signal)
        
        x = torch.stack(signals).to(self.device) # batch_size, 12, num_templates, 450
        x = torch.permute(x, (0,2,1,3)) # batch_size, num_templates, 12, 450
        batch = x.shape[0]
        x = x.reshape(batch * self.max_seq_length, x.shape[2], x.shape[3]) # batch_size*num_templates, 12, 450
        out = self.encoder(x) # batch_size*num_templates, 512

        out = out.reshape(batch, self.max_seq_length, -1) # batch_size, num_templates, 512
        out = out.permute((0,2,1)) # batch_size, 512, seq_length
        # out = self.decoder(out)
        out = torch.stack([torch.hstack([self.class_embeddings, out[i]]) for i in range(len(out))])

        pos_embed = self.positional_embeddings.repeat(len(out), 1, 1)
        pos_embed = pos_embed.permute((0, 2, 1))
        out += pos_embed
        out = out.permute((2, 0, 1))
        out = self.transformer(out)
        out = out.permute((1, 0, 2))

        out = out[:, :27, :]
        demographics = torch.stack([torch.vstack([(torch.mul(self.demo_embed[i], demographics[j])) for i in range(27)]) for j in range(demographics.shape[0])]) 
        out = torch.cat((out, demographics), dim = 2) # batch, 27, time
        out = self.dense(out)
        # batch, 27, 1
        out = torch.squeeze(out, dim=out.ndim-1)

        return out