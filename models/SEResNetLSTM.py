import torch
import torch.nn as nn
import math

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            # nn.ReLU(inplace=True),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)



def convkx1(in_planes, out_planes, kernel_size, stride=1):
    """kx1 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = convkx1(inplanes, planes, kernel_size, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.SiLU(inplace=True)
        self.conv2 = convkx1(planes, planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, kernel_size, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = convkx1(planes, planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.se = SELayer(self.expansion * planes)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.SiLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(.2)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SEResNetLSTM(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2, 2, 2, 2], in_channel=12, out_channel=27, num_additional_features=18, kernel_size=7, zero_init_residual=True):
        super(SEResNetLSTM, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.SiLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], kernel_size)
        self.layer2 = self._make_layer(block, 128, layers[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], kernel_size, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], kernel_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.lstm = nn.LSTM(input_size=512 * block.expansion, hidden_size=150, num_layers=2, batch_first=True, dropout=0.1, bidirectional=True)
        self.fc1 = nn.Linear(num_additional_features, num_additional_features)
        self.fc = nn.Linear(300 + num_additional_features, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        """
    def _make_layer(self, block, planes, blocks, kernel_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, kernel_size, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, kernel_size))

        return nn.Sequential(*layers)



    def forward(self, x, demographics):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        x = x.permute((0, 2, 1))
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0][:, -1, :]
        ag = self.fc1(demographics)
        x = torch.cat((x, ag), dim=1)
        x = self.fc(x)
        return x