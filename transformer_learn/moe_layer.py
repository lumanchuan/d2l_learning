import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, d_ff, output_dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, output_dim)
        )
    def forward(self, x):
        return self.layer(x)


class GatingNetwork(nn.Module):
    def __init__(self, d_model, num_experts, shared_experts, k=2, noise_std=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.shared_experts = shared_experts
        self.k = k  # Top-k selection
        self.noise_std = noise_std  # Noise standard deviation
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        logits = self.gate(x)  # Compute expert scores
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std  # Add noise during training
            logits += noise
        weights = F.softmax(logits, dim=-1)  # Convert to probabilities
        topk_weights, topk_indices = torch.topk(weights, self.k, dim=-1)  # Select top-k experts
        return topk_weights, topk_indices, weights


class MoE(nn.Module):
    def __init__(self, d_model, d_ff, output_dim, num_experts=6, shared_experts=2, k=2, noise_std=0.1):
        super().__init__()
        self.k = k
        self.shared_expert_modules = nn.ModuleList([Expert(d_model, d_ff, output_dim) for _ in range(shared_experts)])
        self.specific_expert_modules = nn.ModuleList([Expert(d_model, d_ff, output_dim) for _ in range(num_experts - shared_experts)])
        self.gating_network = GatingNetwork(d_model, num_experts, shared_experts, k, noise_std)
        self.num_experts = num_experts

    def forward(self, x):
        batch_size = x.shape[0]
        topk_weights, topk_indices, all_weights = self.gating_network(x)

        # Compute shared expert outputs
        shared_outputs = torch.zeros(batch_size, self.shared_expert_modules[0].layer[-1].out_features, device=x.device)
        for i in range(len(self.shared_expert_modules)):
            shared_outputs += self.shared_expert_modules[i](x)
        shared_outputs /= len(self.shared_expert_modules)  # Normalize shared expert contribution

        # Compute task-specific expert outputs
        specific_outputs = torch.zeros(batch_size, self.specific_expert_modules[0].layer[-1].out_features, device=x.device)
        for i in range(self.k):
            expert_idx = topk_indices[:, i] - len(self.shared_expert_modules)  # Offset for specific experts
            expert_weight = topk_weights[:, i].unsqueeze(-1)
            valid_mask = expert_idx >= 0  # Ensure valid expert selection

            expert_outputs = torch.zeros_like(specific_outputs)
            for j, idx in enumerate(expert_idx):
                if valid_mask[j]:
                    expert_outputs[j] = self.specific_expert_modules[idx](x[j:j + 1])

            specific_outputs += expert_weight * expert_outputs.squeeze(1)

        output = shared_outputs + specific_outputs  # Combine shared and task-specific expert outputs
        return output, all_weights

    def load_balancing_loss(self, all_weights):
        """
        Computes a load balancing loss to encourage equal usage of experts.
        """
        expert_probs = all_weights.mean(dim=0)  # Average over batch
        load_balancing_loss = (expert_probs * torch.log(expert_probs + 1e-10)).sum()
        return -load_balancing_loss  # Negative entropy for balance


# 示例用法
if __name__ == "__main__":
    model = MoE(d_model=10, d_ff=512, output_dim=5, num_experts=6, shared_experts=2, k=2, noise_std=0.1)
    x = torch.rand(4, 10)  # 4个样本，输入维度为10
    output, all_weights = model(x)
    loss = model.load_balancing_loss(all_weights)
    print(output.shape)  # 输出应为 (4, 5)
    print("Load balancing loss:", loss.item())