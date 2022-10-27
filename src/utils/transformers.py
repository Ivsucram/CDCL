from re import X
import torch
from torch.nn import Module, ModuleList, Linear, Dropout, LayerNorm, Identity, Parameter, Conv1d, init
import torch.nn.functional as F
from .stochastic_depth import DropPath


class Attention(Module):
    def __init__(self, 
                 dim,
                 num_heads=8,
                 attention_dropout=0.1,
                 projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = Linear(dim, dim * 3, bias=False)
        self.attn_drop = Dropout(attention_dropout)
        self.proj = Linear(dim, dim)
        self.proj_drop = Dropout(projection_dropout)

    def forward(self, x, x2=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if x2 is not None:
            qkv = self.qkv(x2).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            _, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionCrossAttention(Attention):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.0,
                 projection_dropout=0.0,
                 tasks=1):
        super(AttentionCrossAttention, self).__init__(dim, num_heads, attention_dropout, projection_dropout)
        assert tasks > 0, 'The number of tasks should be greater than zero'
        if qk_scale is not None:
            self.scale = qk_scale

        self.n_tasks = tasks

        self.k = [None] * self.n_tasks
        for i in range(self.n_tasks):
            self.k[i] = Linear(dim, dim, bias=True).cuda() # FIXME: Remvoe this cuda() call

        self.qv = Linear(dim, dim * 2, bias=qkv_bias)
        self.attn = None

    def forward(self, x, x2 = None, use_attn=True, task=0):
        B, N, C = x.shape
        if x2 is None:
            qkv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k = self.k[task](x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, v = qkv[0], qkv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            self.attn = attn
            attn = self.attn_drop(attn)

            x = ( attn @ v ) if use_attn else v

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
        else:
            qkv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k = self.k[task](x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, v = qkv[0], qkv[1]

            qkv2 = self.qv(x2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k2 = self.k[task](x2).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q2, v2 = qkv2[0], qkv2[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
            attn3 = (q @ k2.transpose(-2, -1)) * self.scale

            attn = attn.softmax(dim=-1)
            attn2 = attn2.softmax(dim=-1)
            attn3 = attn3.softmax(dim=-1)
            self.attn = attn
            attn = self.attn_drop(attn)
            attn2 = self.attn_drop(attn2)
            attn3 = self.attn_drop(attn3)

            x = ( attn @ v ) if use_attn else v
            x2 = ( attn2 @ v2 ) if use_attn else v2
            x3 = ( attn3 @ v2 ) if use_attn else v2

            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)

            x2 = x2.transpose(1, 2).reshape(B, N, C)
            x2 = self.proj(x2)
            x2 = self.proj_drop(x2)

            x3 = x3.transpose(1, 2).reshape(B, N, C)
            x3 = self.proj(x3)
            x3 = self.proj_drop(x3)

        return x, x2, x3, None
        


class TransformerCrossEncoderLayer(Module):
    def __init__(self,
                 dim,
                 num_heads,
                 dim_feedforward=2048,
                 dropout=0.0,
                 attention_dropout=0.0,
                 drop_path_rate=0.0,
                 tasks=1):
        super().__init__()
        assert tasks > 0, 'The number of tasks should be greater than zero'
        self.pre_norm = LayerNorm(dim)
        self.self_attn = AttentionCrossAttention(dim, num_heads=num_heads, attention_dropout=attention_dropout, projection_dropout=drop_path_rate, tasks=tasks)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else Identity()
        self.norm1 = LayerNorm(dim)

        self.linear1 = Linear(dim, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, dim)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

    def forward(self, x, x2=None, x1_x2_fusion=None):
        if x2 is None or x1_x2_fusion is None:
            xa_attn = self.self_attn(self.pre_norm(x))
            # xa_attn = self.conv_one(xa_attn)
            xa = x + self.drop_path(xa_attn)
            xa = xa + self.drop_path(self._mlp(self.norm1(xa)))
            return xa
        else:
            xa_attn, xa_attn2, xa_attn3, cross_attn = self.self_attn(self.pre_norm(x),self.pre_norm(x2))
            # xa_attn, xa_attn2, xa_attn3, cross_attn = self.conv_one(xa_attn), self.conv_one(xa_attn2), self.conv_one(xa_attn3), self.conv_one(cross_attn)
            xa = x + self.drop_path(xa_attn)
            xa = xa + self.drop_path(self._mlp(self.norm1(xa)))

            xb = x2 + self.drop_path(xa_attn2)
            xb = xb + self.drop_path(self._mlp(self.norm1(xb)))

            xab = x1_x2_fusion + self.drop_path(xa_attn3)
            xab = xab + self.drop_path(self._mlp(self.norm1(xab)))

            return xa, xb, xab, cross_attn
    
    def _mlp(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class TransformerClassifier(Module):
    def __init__(self,
                 seq_pool=True,
                 embedding_dim=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 dropout=0.1,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 positional_embedding='learnable',
                 sequence_length=None,
                 use_attention=True,
                 tasks=1):
        super().__init__()
        assert tasks > 0, 'The number of tasks should be greater than zero'

        positional_embedding = positional_embedding if \
            positional_embedding in ['sine', 'learnable', 'none'] else 'sine'
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.seq_pool = seq_pool
        self.num_tokens = 0
        self.use_attention = use_attention

        assert sequence_length is not None or positional_embedding == 'none', \
            f"Positional embedding is set to {positional_embedding} and" \
            f" the sequence length was not specified."

        if not seq_pool:
            sequence_length += 1
            self.class_emb = Parameter(torch.zeros(1, 1, self.embedding_dim),
                                       requires_grad=True)
            self.num_tokens = 1
        else:
            self.attention_pool = Linear(self.embedding_dim, 1)

        if positional_embedding != 'none':
            if positional_embedding == 'learnable':
                self.positional_emb = Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
                init.trunc_normal_(self.positional_emb, std=0.2)
            else:
                self.positional_emb = Parameter(self.sinusoidal_embedding(sequence_length, embedding_dim),
                                                requires_grad=False)
        else:
            self.positional_emb = None

        self.dropout = Dropout(p=dropout)
        dpr = [x.item() for x in torch.linspace(0, stochastic_depth, num_layers)]
        self.blocks = ModuleList([
            TransformerCrossEncoderLayer(dim=embedding_dim, num_heads=num_heads,
                                     dim_feedforward=dim_feedforward, dropout=dropout,
                                     attention_dropout=attention_dropout, drop_path_rate=dpr[i],
                                     tasks=tasks
            )
            for i in range(num_layers)])
        self.norm = LayerNorm(embedding_dim)

        self.fc_a = Linear(embedding_dim, num_classes)
        
        self.fc_i = [None] * tasks
        for i in range(tasks):
            self.fc_i[i] = Linear(embedding_dim, num_classes // tasks).cuda()
        self.apply(self.init_weight)

    def forward(self, x, x2=None, x_x2_fusion=None, task=0):
        if self.positional_emb is None and x.size(1) < self.sequence_length:
            x = F.pad(x, (0, 0, 0, self.n_channels - x.size(1)), mode='constant', value=0)

        if not self.seq_pool:
            cls_token = self.class_emb.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        if self.positional_emb is not None:
            x += self.positional_emb

        x = self.dropout(x)

        if x2 is not None:
            if self.positional_emb is None and x2.size(1) < self.sequence_length:
                x2 = F.pad(x2, (0, 0, 0, self.n_channels - x2.size(1)), mode='constant', value=0)

            if not self.seq_pool:
                cls_token = self.class_emb.expand(x2.shape[0], -1, -1)
                x2 = torch.cat((cls_token, x2), dim=1)

            if self.positional_emb is not None:
                x2 += self.positional_emb

            x2 = self.dropout(x2)
            x_x2_fusion = x2 if x_x2_fusion is None else x_x2_fusion
            cross_attention_list = []

        for blk in self.blocks:
            if x2 is not None: 
                x, x2, x_x2_fusion, cross_attention = blk(x, x2, x_x2_fusion)
                cross_attention_list.append(cross_attention)
            else:
                x = blk(x)

        x = self.norm(x)

        if x2 is not None:
            x2 = self.norm(x2)
            x_x2_fusion = self.norm(x_x2_fusion)

            return self.out(x, x2, x_x2_fusion, task=task)
        return self.out(x, task=task)

    def out(self, x, x2=None, x_x2_fusion=None, task=0):
        if self.seq_pool:
            x_ = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        else:
            x_ = x[:, 0]
        
        if x2 is not None:
            if self.seq_pool:
                x2_ = torch.matmul(F.softmax(self.attention_pool(x2), dim=1).transpose(-1, -2), x2).squeeze(-2)
                x_x2_fusion_ = torch.matmul(F.softmax(self.attention_pool(x_x2_fusion), dim=1).transpose(-1, -2), x_x2_fusion).squeeze(-2)
            else:
                x2_ = x2[:, 0]
                x_x2_fusion_ = x_x2_fusion[:, 0]

            return ((self.fc_i[task](x_), self.fc_i[task](x2_), self.fc_i[task](x_x2_fusion_)), 
                    (self.fc_a(x_), self.fc_a(x2_), self.fc_a(x_x2_fusion_)), 
                    (x_, x2_, x_x2_fusion_), 
                    (x, x2, x_x2_fusion))
            
        return (self.fc_i[task](x_)), (self.fc_a(x_)), (x_), (x)

    @staticmethod
    def init_weight(m):
        if isinstance(m, Linear):
            init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, Linear) and m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, LayerNorm):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)

    @staticmethod
    def sinusoidal_embedding(n_channels, dim):
        pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                                for p in range(n_channels)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])
        pe[:, 1::2] = torch.cos(pe[:, 1::2])
        return pe.unsqueeze(0)