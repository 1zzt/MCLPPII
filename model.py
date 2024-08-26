import copy
import math

# import ml_collections
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import reduce
from torch.nn.modules.utils import _triple
import numpy as np
from debertav2 import DebertaV2Model


class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList(
            [nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)]
        )
        self.linear = nn.ModuleList(
            [nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)]
        )

    def forward(self, x):
        pre_x = x
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = dim**-0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g



def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.normal_(m.bias.data, std=1e-6)


class ModalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModalEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # RMSNorm(output_size),
            nn.ELU(),
            # Highway(1, output_size),
            nn.Linear(hidden_size, output_size),
            # RMSNorm(hidden_size),
            nn.ELU(),
            # nn.Linear(hidden_size,hidden_size),
            # RMSNorm(hidden_size),
            # nn.GELU(),
            # nn.Linear(hidden_size,output_size),
            # RMSNorm(output_size),
            # nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Projector(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Projector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self, input_size):
        super(Predictor, self).__init__()
        self.net = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.net(x)


class PPIMMultiModalNet(nn.Module):
    def __init__(self):
        super(PPIMMultiModalNet, self).__init__()

        self.drug_encoder = DebertaV2Model.from_pretrained(
            "/home/zitong/project/PPIM/11_contrastive/DeBERTA/pretrainV2/output_dir"
        )
        for param in self.drug_encoder.parameters():
            param.requires_grad = False

        encoder_hidden_size = 256
        encoder_output_size = 256
        self.bert_encoder = ModalEncoder(768, encoder_hidden_size, encoder_output_size)
        self.ecfp_encoder = ModalEncoder(1024, encoder_hidden_size, encoder_output_size)
        self.hashap_encoder = ModalEncoder(1024, encoder_hidden_size, encoder_output_size)
        self.rdk_encoder = ModalEncoder(1024, encoder_hidden_size, encoder_output_size)

        proj_hidden_size = 1024
        proj_size = 256
        self.bert_projector = Projector(encoder_output_size, proj_hidden_size, proj_size)
        self.ecfp_projector = Projector(encoder_output_size, proj_hidden_size, proj_size)
        self.hashap_projector = Projector(encoder_output_size, proj_hidden_size, proj_size)
        self.rdk_projector = Projector(encoder_output_size, proj_hidden_size, proj_size)

        self.predict_fusion = Predictor(proj_size * 4)

        self.temperature = 0.07
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def graph_construct_per_sample(self, modality_feature):  # 4*B*H    modality_feature: torch.Size([4, B, 256])
        devices = modality_feature.get_device()
        batch_modality_graph = torch.zeros(
            [
                modality_feature.shape[1],
                modality_feature.shape[0],
                modality_feature.shape[0],
            ]
        ).to(devices)   # batch_modality_graph: torch.Size([B, 4, 4])
        diag_mask = torch.zeros([modality_feature.shape[1], modality_feature.shape[0]]).to(devices)     # torch.Size([B, 4])
        for b_i in range(modality_feature.shape[1]):    # 计算每个样本的不同特征之间的余弦相似度
            group_modality = modality_feature[:, b_i, :]    #  torch.Size([4, 256])
            group_sim = F.cosine_similarity(        # torch.Size([4, 4])
                group_modality.unsqueeze(1),    # torch.Size([4, 1, 256])
                group_modality.unsqueeze(0),    # torch.Size([1, 4, 256])
                dim=2,
            )
            revised_modality_graph = group_sim * F.softmax(group_sim, dim=-1)   # 对余弦相似度进行加权  [4, 4]
            group_sim = F.normalize(revised_modality_graph, dim=-1)     # 归一化，使得每一行的和为1 [4, 4]
            batch_modality_graph[b_i, :, :] = group_sim     # batch_modality_graph: torch.Size([B, 4, 4])
            diag_mask[b_i, :] = torch.diag(group_sim)

        # 每个batch里三个fp间的similarity(B*4*4),
        # 每个batch里每个fp自己和自己的similarity(B*4), 因为revised过所以不为1
        return batch_modality_graph, diag_mask

    def forward(self, input_id, att_mask, hashap, ecfp, rdkfp):
        outputs = self.drug_encoder(input_id, att_mask, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states
        first_hidden = hidden_states[0]
        last_hidden = hidden_states[-1]

        bert_features = ((first_hidden + last_hidden) / 2.0 * att_mask.unsqueeze(-1)).sum(
            1
        ) / att_mask.sum(-1).unsqueeze(-1)
        bert_features = self.bert_encoder(bert_features)
        ecfp_features = self.ecfp_encoder(ecfp)
        hashap_features = self.hashap_encoder(hashap)
        rdk_features = self.rdk_encoder(rdkfp)

        bert_features = self.bert_projector(bert_features)
        ecfp_features = self.ecfp_projector(ecfp_features)
        hashap_features = self.hashap_projector(hashap_features)
        rdk_features = self.rdk_projector(rdk_features)

        # 计算CL损失
        # CLR_loss = fp间两两计算infoNCELoss后加权求平均
        # 对应权重为每个mol的fp间的相似度，即每个mol的权重都不同
        concat_feature = torch.cat(
            (
                ecfp_features.unsqueeze(0),
                bert_features.unsqueeze(0),
                hashap_features.unsqueeze(0),
                rdk_features.unsqueeze(0),
            ),
            dim=0,
        )       # torch.Size([4, B, 256])
        graph, modality_weight = self.graph_construct_per_sample(concat_feature)

        CLR_loss = 0
        cnt = 0
        batch_size = rdk_features.size(0)
        all_features = [
            ecfp_features,  # b, 256
            bert_features,
            hashap_features,
            rdk_features,
        ]   
        for i in range(len(all_features)):      # all_features: [4, b, 256]
            for j in range(i + 1, len(all_features)):   # 所有样本的第i种特征和第j种特征
                features = torch.cat([all_features[i], all_features[j]], dim=0)     # torch.Size([2*B, 256])
                logits, labels = self.info_nce_loss(features, batch_size, n_views=2)    # logits: 2B*(2B-1); label 全0, B
                # 加权  graph是权重: torch.Size([32, 4, 4])每个样本4种特征之间的余弦相似度
                logits = (  
                    torch.cat(  # torch.Size([64, 1])横着的都是同一个数
                        (   # graph[:, i, j] 所有样本第i种特征和第j种特征之间的余弦相似度 torch.Size([32])
                            graph[:, i, j].reshape(len(all_features[0]), 1),    # torch.Size([32, 1]), graph[:, i, j]: torch.Size([32])
                            graph[:, i, j].reshape(len(all_features[0]), 1),    # torch.Size([32, 1])
                        )
                    )
                    .expand(logits.shape[0], logits.shape[1])   # torch.Size([64, 63])
                    .mul(logits)    # 按位相乘
                )
                CLR_loss += self.loss_fct(logits, labels)
                cnt += 1
        CLR_loss /= cnt

        # 每个fp的特征乘上权重
        pred_all_features = [
            ecfp_features,
            bert_features,
            hashap_features,
            rdk_features,
        ]
        final_features = torch.cat(
            (
                modality_weight[:, 0]
                .unsqueeze(1)
                .expand(modality_weight.shape[0], pred_all_features[0].shape[1])
                * pred_all_features[0],
                modality_weight[:, 1]
                .unsqueeze(1)
                .expand(modality_weight.shape[0], pred_all_features[0].shape[1])
                * pred_all_features[1],
                modality_weight[:, 2]
                .unsqueeze(1)
                .expand(modality_weight.shape[0], pred_all_features[0].shape[1])
                * pred_all_features[2],
                modality_weight[:, 3]
                .unsqueeze(1)
                .expand(modality_weight.shape[0], pred_all_features[0].shape[1])
                * pred_all_features[3],
            ),
            dim=1,
        )

        # np.savetxt('featureout/' + bromodomain_histone + '/tsne-seed-2.csv', test_save_np, delimiter=',')
        fusion_pred = self.predict_fusion(final_features)
        return fusion_pred, CLR_loss

    def compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def compute_l2_loss(self, w):
        return torch.square(w).sum()

    def info_nce_loss(self, features, batch_size, n_views):     # 所有样本的第i种特征和第j种特征
        devices = features.get_device()
        labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(devices) # B*B   
        # labels looks like & is B*B
        # ---                               ---
        # | 1, 0, 0, ... ..., 1, 0, 0, ... ...| [样本1标签，样本1标签]
        # | 0, 1, 0, ... ..., 0, 1, 0, ... ...| [样本2标签，样本2标签]
        # | 0, 0, 1, ... ..., 0, 0, 1, ... ...| [样本3标签，样本3标签]
        # | .                                .|
        # | .                                .|
        # | .                                .|
        # ---                               ---
        features = F.normalize(features, dim=1, p=2)    # torch.Size([B, 256])
        # 内积
        similarity_matrix = torch.matmul(features, features.T)  # torch.Size([B, B]) # 计算第i种特征，两两样本之间的相似度、同一个样本第i种特征和第j种特征的相似度、不同样本第i种特征和第j种特征的相似度

        # discard the main diagonal from both: labels and similarities matrix   创建一个对角线为True（或1），其余元素为False（或0）的布尔掩码矩阵
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(devices) 

        # B*(B-1) 移除对角线上的元素,删除label中第一种特征标签
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # B*1   pos：每个样本第i种特征和第j种特征的相似性
        positives = similarity_matrix[labels.bool()].view(similarity_matrix.shape[0], -1)
        # B*(B-2)   neg: 不同样本第i种特征和第i种特征+不同样本第i种特征和第j种特征
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # B*(B-1)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(devices)

        # logits = logits / self.temperature
        return logits, labels


def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix
