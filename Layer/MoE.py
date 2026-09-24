import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """单个专家网络"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MoE(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2, mlp_ratio=4.0, drop=0.3, gate_temp=2.5):
        super().__init__()
        self.expert_activation_counts = None
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate_temp = gate_temp
        self.hidden_dim = int(dim * mlp_ratio)
        self.experts = nn.ModuleList([
            Expert(dim, self.hidden_dim, dim, drop=drop)
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        B, T, D = x.shape
        flat_x = x.flatten(0, 1)  # [N, D], 其中 N = B*T
        N = flat_x.shape[0]

        # 门控计算
        gate_logits = self.gate(flat_x) / self.gate_temp
        gate_probs = F.softmax(gate_logits, dim=1)  # [N, num_experts]

        # 计算辅助损失 (Load Balancing Loss)
        # 计算 Importance: 每个专家在所有 token 上的平均概率
        importance = gate_probs.mean(dim=0)  # [num_experts]

        # 计算 Load: 每个专家被选中的频率
        _, top_k_indices = torch.topk(gate_probs, self.top_k, dim=1)  # [N, top_k]
        # 创建一个 one-hot 掩码，表示哪些专家被选中了
        expert_mask = torch.zeros(N, self.num_experts, device=x.device)
        expert_mask.scatter_(1, top_k_indices, 1.0)
        load = expert_mask.mean(dim=0)  # [num_experts]

        # 辅助损失公式: num_experts * sum(importance * load)
        # 当分布完全均匀时，该值达到最小
        aux_loss = self.num_experts * torch.sum(importance * load)

        top_k_weights = gate_probs.gather(dim=1, index=top_k_indices)
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=1, keepdim=True).clamp(min=1e-5))
        output = torch.zeros_like(flat_x)

        for expert_idx in range(self.num_experts):
            mask = (top_k_indices == expert_idx)
            if not mask.any(): continue
            element_indices, k_indices = mask.nonzero(as_tuple=True)
            expert_input = flat_x[element_indices]
            expert_output = self.experts[expert_idx](expert_input)
            weights = top_k_weights[element_indices, k_indices].unsqueeze(1)
            output[element_indices] += weights * expert_output

        return output.view(B, T, D), aux_loss


class ProgressiveMoE(nn.Module):
    """Progressive MoE with variable expert count and shared expert.

    When num_experts=0: falls back to shared expert only (dense MLP).
    Shared expert bypasses the gate and is always added to all tokens.
    """

    def __init__(self, dim, num_experts=0, shared_expert=False, top_k=2,
                 mlp_ratio=4.0, drop=0.3, gate_temp=2.5):
        super().__init__()
        self.num_experts = num_experts
        self.shared_expert = shared_expert
        self.top_k = min(top_k, num_experts) if num_experts > 0 else 0
        self.gate_temp = gate_temp
        self.hidden_dim = int(dim * mlp_ratio)

        if num_experts > 0:
            self.experts = nn.ModuleList([
                Expert(dim, self.hidden_dim, dim, drop=drop)
                for _ in range(num_experts)
            ])
            self.gate = nn.Linear(dim, num_experts)

        if shared_expert:
            self.shared = Expert(dim, self.hidden_dim, dim, drop=drop)

    def forward(self, x):
        B, T, D = x.shape
        flat_x = x.flatten(0, 1)
        N = flat_x.shape[0]

        if self.num_experts > 0:
            gate_logits = self.gate(flat_x) / self.gate_temp
            gate_probs = F.softmax(gate_logits, dim=1)

            importance = gate_probs.mean(dim=0)
            _, top_k_indices = torch.topk(gate_probs, self.top_k, dim=1)
            expert_mask = torch.zeros(N, self.num_experts, device=x.device)
            expert_mask.scatter_(1, top_k_indices, 1.0)
            load = expert_mask.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(importance * load)

            top_k_weights = gate_probs.gather(dim=1, index=top_k_indices)
            top_k_weights = top_k_weights / (top_k_weights.sum(dim=1, keepdim=True).clamp(min=1e-5))

            output = torch.zeros_like(flat_x)
            for expert_idx in range(self.num_experts):
                mask = (top_k_indices == expert_idx)
                if not mask.any():
                    continue
                element_indices, k_indices = mask.nonzero(as_tuple=True)
                expert_input = flat_x[element_indices]
                expert_output = self.experts[expert_idx](expert_input)
                weights = top_k_weights[element_indices, k_indices].unsqueeze(1)
                output[element_indices] += weights * expert_output
        else:
            aux_loss = torch.tensor(0.0, device=x.device)
            output = torch.zeros_like(flat_x)

        if self.shared_expert:
            output = output + self.shared(flat_x)

        return output.view(B, T, D), aux_loss
