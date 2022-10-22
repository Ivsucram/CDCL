import torch.nn as nn
from .tokenizer import Tokenizer

class Shrink(nn.Module):
    def __init__(self,
                 kernel_size, stride, padding,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=1,
                 n_input_channels=3,
                 n_output_channels=768,
                 activation=None,
                 max_pool=True,
                 conv_bias=False,
                 img_size=16):
        super(Shrink, self).__init__()
        self.unflatten = nn.Unflatten(-1,(img_size,img_size))
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=n_output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False)
        
    def forward(self, x, x2=None, x1_x2_fusion=None):
      x = x.transpose(-2,-1)
      x = self.unflatten(x)
      x = self.tokenizer(x)

      if x2 is not None:
        x2 = x2.transpose(-2,-1)
        x2 = self.unflatten(x2)
        x2 = self.tokenizer(x2)

        if x1_x2_fusion is not None:
            x1_x2_fusion = x1_x2_fusion.transpose(-2,-1)
            x1_x2_fusion = self.unflatten(x2)
            x1_x2_fusion = self.tokenizer(x2)
        
            return x, x1_x2_fusion, x2
        return x, x2,
      return x