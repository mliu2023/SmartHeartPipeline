import torch.nn as nn
from math import floor
def conv_output_shape(w, kernel_size=1, stride=1, pad=0, dilation=1):
    w = floor(((w + (2 * pad) - (dilation * (kernel_size - 1)) - 1)/ stride) + 1)
    return w

def cnn_output_shape(model, input_length):
    output_length = input_length
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            kernel_size = m.kernel_size[0]
            stride = m.stride[0]
            if(kernel_size != 1):
                pad = m.padding[0]
                dilation = m.dilation[0]
                output_length = conv_output_shape(output_length, kernel_size, stride, pad, dilation)
        elif isinstance(m, nn.MaxPool1d):
            kernel_size = m.kernel_size
            stride = m.stride
            if(kernel_size != 1):
                pad = m.padding
                dilation = m.dilation
                output_length = conv_output_shape(output_length, kernel_size, stride, pad, dilation)
    return output_length

def config2_output_shape(module, input_length):
    output_length = input_length
    for m in module.modules():
        if isinstance(m, nn.MaxPool1d):
            kernel_size = m.kernel_size
            stride = m.stride
            if(stride != 1):
                pad = m.padding
                dilation = m.dilation
                output_length = conv_output_shape(output_length, kernel_size, stride, pad, dilation)
    return output_length