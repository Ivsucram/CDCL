from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
from .utils.transformers import TransformerClassifier
from .utils.tokenizer import Tokenizer
from .utils.shink import Shrink
from .utils.helpers import pe_check, fc_check

try:
    from timm.models.registry import register_model
except ImportError:
    from .registry import register_model

class CDCL(nn.Module):
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
                 positional_embedding='learnable',
                 tasks=1,
                 *args, **kwargs):
        super(CDCL, self).__init__()

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
            num_classes=2048,
            positional_embedding=positional_embedding,
            tasks=tasks
        )

        self.identity = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, '8192-8192-8192'.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            # layers.append(nn.BatchNorm1d(sizes[i + 1]))  # I commented this because I am not sending a batch of samples, but a single image per time - got lazy, sorry
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2, task=None):
        z1 = self.projector(self.identity(self.classifier(self.tokenizer(y1))[0]))
        z2 = self.projector(self.identity(self.classifier(self.tokenizer(y2))[0]))

        # empirical cross-correlation matrix
        # c = self.bn(z1).T @ self.bn(z2) # I commented this because I am not sending a batch of samples, but a single image per time - got lazy, sorry
        c = z1.T @ z2

        # sum the cross-correlation matrix between all gpus
        # c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c) # Only used when we are running the code in multiple GPUs

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + 0.005 * off_diag # selg.args.lambd = 0.005
        return loss

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _cdcl(arch, pretrained, progress,
         num_layers, num_heads, mlp_ratio, embedding_dim,
         n_input_channels=3, kernel_size=3, stride=None, padding=None,
         positional_embedding='learnable',
         *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))
    model = CDCL(n_input_channels=n_input_channels,
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


def cdcl_7(arch, pretrained, progress, *args, **kwargs):
    return _cdcl(arch, pretrained, progress, num_layers=7, num_heads=4, mlp_ratio=2, embedding_dim=256,
                *args, **kwargs)


def cdcl_14(arch, pretrained, progress, *args, **kwargs):
    return _cdcl(arch, pretrained, progress, num_layers=14, num_heads=6, mlp_ratio=3, embedding_dim=384,
                *args, **kwargs)


@register_model
def clcd_7_7x2_28(pretrained=False, progress=False,
                  img_size=28, positional_embedding='learnable',
                  *args, **kwargs):
    return cdcl_7('clcd_7_7x2_28', pretrained, progress,
                 n_input_channels=1, kernel_size=7, n_conv_layers=2,
                 img_size=img_size, positional_embedding=positional_embedding,
                 *args, **kwargs)

@register_model
def clcd_14_7x2_224(pretrained=False, progress=False,
                   img_size=224, positional_embedding='learnable',
                   *args, **kwargs):
    return cdcl_14('clcd_14_7x2_224', pretrained, progress,
                  kernel_size=7, n_conv_layers=2,
                  img_size=img_size, positional_embedding=positional_embedding,
                  *args, **kwargs)
