from torch.hub import load_state_dict_from_url
import torch.nn as nn
from .utils.transformers import TransformerClassifier
from .utils.tokenizer import Tokenizer
from .utils.shink import Shrink
from .utils.helpers import pe_check, fc_check

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

class CLCD(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 n_conv_layers=1,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 tasks=1,
                 *args, **kwargs):
        super(CLCD, self).__init__()

        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   max_pool=True,
                                   activation=nn.ReLU,
                                   n_conv_layers=n_conv_layers,
                                   conv_bias=False
        )

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding,
            tasks=tasks
        )

    def forward(self, x, x2=None, task=0):
        x = self.tokenizer(x)

        if x2 is not None:
            x2 = self.tokenizer(x2)
            (ix, ix2, ix_x2), (ax, ax2, ax_x2), (feat_x, feat_x2, feat_x_x2), (attn_x, attn_x2, attn_x_x2), (pkw, pkb), (kw, kb) = self.classifier(x, x2, task=task)
            return (ix, ix2, ix_x2), (ax, ax2, ax_x2), (feat_x, feat_x2, feat_x_x2), (pkw, pkb), (kw, kb)

        (ix), (ax), (feat_x), (attn_x), (pkw, pkb), (kw, kb)  = self.classifier(x, task=task)
        return (ix), (ax), (feat_x), (pkw, pkb), (kw, kb)


def _clcd(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         n_input_channels=3, kernel_size=3, stride=None, padding=None,
         positional_embedding='learnable',
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CLCD(n_input_channels=n_input_channels,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                tasks=kwargs['args'].tasks,
                *args, **kwargs)

    return model


def clcd_7(arch, pretrained, progress, *args, **kwargs):
    return _clcd(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def clcd_14(arch, pretrained, progress, *args, **kwargs):
    return _clcd(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


@register_model
def clcd_7_7x2_28(pretrained=False, progress=False,
                  img_size=28, positional_embedding='learnable', num_classes=102,
                  *args, **kwargs):
    return clcd_7('clcd_7_7x2_28', pretrained, progress,
                 n_input_channels=1, kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 num_classes=num_classes,
                 *args, **kwargs)

@register_model
def clcd_14_7x2_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable', num_classes=1000,
                   *args, **kwargs):
    return clcd_14('clcd_14_7x2_224', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  num_classes=num_classes,
                  *args, **kwargs)
