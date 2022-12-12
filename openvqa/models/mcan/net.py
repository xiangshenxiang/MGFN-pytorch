# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------
from collections import OrderedDict
from openvqa.utils1.make_mask import make_mask
from openvqa.ops.fc import FC, MLP
from openvqa.ops.layer_norm import LayerNorm
from openvqa.models.mcan.mca import MCA_ED
from openvqa.models.mcan.adapter import Adapter

import torch.nn as nn
import torch.nn.functional as F
import torch
def exist(x):
    return x is not None


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, len_sen):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.proj = nn.Conv1d(len_sen, len_sen, 1)

        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        res, gate = torch.chunk(x, 2, -1)  # bs,n,d_ff
        ###Norm
        gate = self.ln(gate)  # bs,n,d_ff
        ###Spatial Proj
        gate = self.proj(gate)  # bs,n,d_ff

        return res * gate








class GluLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        # 第一个线性层
        self.fc1 = nn.Linear(input_size, output_size)
        # 第二个线性层
        self.fc2 = nn.Linear(input_size, output_size)
        # pytorch的GLU层
        self.glu = nn.GLU()

    def forward(self, x):
        # 先计算第一个线性层结果
        a = self.fc1(x)
        # 再计算第二个线性层结果
        b = self.fc2(x)
        # 拼接a和b，水平扩展的方式拼接
        # 然后把拼接的结果传给glu
        return self.glu(torch.cat((a, b), dim=1))
    # ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------
class gMLP(nn.Module):
    def __init__(self,  len_sen=14, dim=512, d_ff=1024, num_layers=1):
        super().__init__()
        self.num_layers = num_layers


        self.gmlp = nn.ModuleList([Residual(nn.Sequential(OrderedDict([
            ('ln1_%d' % i, nn.LayerNorm(dim)),
            ('fc1_%d' % i, nn.Linear(dim, d_ff * 2)),
            ('gelu_%d' % i, nn.GELU()),
            ('sgu_%d' % i, SpatialGatingUnit(d_ff, len_sen)),
            ('fc2_%d' % i, nn.Linear(d_ff, dim)),
        ]))) for i in range(num_layers)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.Softmax(-1)
        )
    def forward(self, x):
        # embedding

        # gMLP
        y = nn.Sequential(*self.gmlp)(x)

        # to logits
        logits = self.to_logits(y)

        return logits
class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.__C = __C

        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )
        self.gmlp = gMLP(len_sen=14, dim=__C.WORD_EMBED_SIZE, d_ff=1024)
        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.gru = nn.GRU(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.adapter = Adapter(__C)

        self.backbone = MCA_ED(__C)

        # Flatten to vector
        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.gate1 = nn.Linear(__C.HIDDEN_SIZE * 4, 1)
        # self.gate1 = nn.Linear(__C.HIDDEN_SIZE * 2, 1)
        self.gate2 = nn.Linear(__C.HIDDEN_SIZE * 2, 1)
        # self.gate1 = nn.Linear(__C.HIDDEN_SIZE * 1, 1)
        # Classification layers
        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        self.GLULAyer = GluLayer(__C.FLAT_OUT_SIZE, answer_size)

    def forward(self, frcn_feat, grid_feat, bbox_feat, ques_ix):
        # ques_ix 32*14 bbox_feat 32*100*4 grid_feat 32*1

        # Pre-process Language Feature
        lang_feat_mask = make_mask(ques_ix.unsqueeze(2))
        lang_feat = self.embedding(ques_ix)#出来32*14*300
        lang_feat, _ = self.lstm(lang_feat) #出来14*512
        # lang_feat = self.gmlp(lang_feat)

        img_feat, img_feat_mask = self.adapter(frcn_feat, grid_feat, bbox_feat)

        # Backbone Framework
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )

        # Flatten to vector
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )#32 1024

        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )#32 1024

        # sum_feat = torch.cat((lang_feat, img_feat), dim=1)  # (64,3072) 32.2048
        #
        # gate = F.sigmoid(self.gate2(self.gate1(sum_feat)))  # (64,1)    32.1
        # proj_feat = lang_feat + torch.mul(img_feat, gate)   # (64,1024)  32 1024
        #门控
        sum_feat = torch.cat((lang_feat, img_feat), dim=1)#WTV[T,V] 32,2048
        VT = F.sigmoid(self.gate1(sum_feat))#VVT 32,1
        # VM = F.sigmoid(self.gate2(img_feat))#Vm  32,1


        img_feat = torch.mul(img_feat, VT)

        lang_feat_fanshu = torch.norm(lang_feat,p=2)
        img_feat_fanshu = torch.norm(img_feat,p=2)
        gama = min(lang_feat_fanshu/img_feat_fanshu,1)
        # print("mark333",lang_feat_fanshu/img_feat_fanshu)
        proj_feat = lang_feat + gama * img_feat

        # addon = self.calcAddon(torch.cat([covarepState, facetState], 1))  # h
        #
        # addonL2 = torch.norm(addon, 2, 1)
        # addonL2 = torch.max(addonL2, torch.tensor([1.0]).to(gc.device)) / torch.tensor([gc.shift_weight]).to(gc.device)
        # addon = addon / addonL2.unsqueeze(1)
        # addon = addon.data.contiguous().view(batch, gc.padding_len, gc.wordDim)
        #
        # wordsL2 = torch.norm(words, 2, 2).unsqueeze(2)
        # wordInput = self.dropWord(words + addon * wordsL2)
        #
        # Classification layers
        # proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)


        proj_feat = self.proj(proj_feat)
        # proj_feat = self.GLULAyer(proj_feat)
        return proj_feat

