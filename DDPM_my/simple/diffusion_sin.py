import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random


# 简单的噪声预测网络
class SimpleNoisePredictor(nn.Module):
    def __init__(self, seq_length, hidden_dim=64):
        super().__init__()
        self.seq_length = seq_length
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 主要网络
        self.net = nn.Sequential(
            nn.Linear(seq_length + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, seq_length)
        )

    def forward(self, x, t):
        # x: [batch_size, seq_length]
        # t: [batch_size]
        t = t.unsqueeze(-1).float()  # [batch_size, 1]
        t_emb = self.time_mlp(t)  # [batch_size, hidden_dim]

        # 连接输入和时间嵌入
        x_flat = torch.cat([x, t_emb], dim=-1)  # [batch_size, seq_length + hidden_dim]

        # 预测噪声
        noise_pred = self.net(x_flat)  # [batch_size, seq_length]
        return noise_pred


# 扩散模型
class DenoiseDiffusion:
    def __init__(self, n_steps, device):
        self.n_steps = n_steps
        self.device = device

        # 定义 beta
        self.beta = torch.linspace(1e-4, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x_0, t, noise=None):
        """向干净数据添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_t = self.alpha_bar[t]
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)[:, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - alpha_bar_t)[:, None]

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise

    def p_sample(self, eps_model, x_t, t):
        """去噪步骤"""
        alpha_t = self.alpha[t][:, None]
        alpha_bar_t = self.alpha_bar[t][:, None]

        eps_theta = eps_model(x_t, t)
        mean = (1. / torch.sqrt(alpha_t)) * (x_t - (1. - alpha_t) / torch.sqrt(1. - alpha_bar_t) * eps_theta)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(self.beta[t])[:, None]
            return mean + sigma_t * noise
        else:
            return mean

    def loss(self, eps_model, x_0):
        """计算损失"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)

        noise = torch.randn_like(x_0)
        x_t, target = self.q_sample(x_0, t, noise)
        eps_theta = eps_model(x_t, t)

        return torch.nn.functional.mse_loss(eps_theta, target)


def generate_sin_data(batch_size, seq_length, noise_scale):
    """生成正弦波训练数据"""
    x = np.linspace(0, 2 * np.pi, seq_length)
    data = []
    for _ in range(batch_size):
        y = np.sin(x) + np.random.normal(0, noise_scale, seq_length)
        data.append(y)
    return torch.tensor(data, dtype=torch.float32)

def generate_sin_cos_data(batch_size, seq_length, noise_scale):
    """生成正弦波训练数据"""
    x = np.linspace(0, 2 * np.pi, seq_length)
    data = []
    for _ in range(batch_size):
        y = np.sin(x) + np.cos(x) + np.random.normal(0, noise_scale, seq_length)
        data.append(y)
    return torch.tensor(data, dtype=torch.float32)

def generate_cos_data(batch_size, seq_length, noise_scale):
    """生成正弦波训练数据"""
    x = np.linspace(0, 2 * np.pi, seq_length)
    data = []
    for _ in range(batch_size):
        y = np.cos(x) + np.random.normal(0, noise_scale, seq_length)
        data.append(y)
    return torch.tensor(data, dtype=torch.float32)


def main():
    # 设置参数
    seq_length = 64
    batch_size = 32
    n_steps = 10000
    n_epochs = 1_000_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_name = f'results_step{n_steps}_epoch{n_epochs}'

    # 创建保存结果的文件夹
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # 初始化模型
    eps_model = SimpleNoisePredictor(seq_length=seq_length).to(device)
    diffusion = DenoiseDiffusion(n_steps=n_steps, device=device)
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(n_epochs):
        # 生成训练数据
        # x_0 = generate_sin_data(batch_size, seq_length, 0.3).to(device)
        x_0 = generate_sin_data(batch_size, seq_length, 0.).to(device)

        # 训练步骤
        optimizer.zero_grad()
        loss = diffusion.loss(eps_model, x_0)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10000 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            # 生成样本
            with torch.no_grad():
                # 去噪起点为纯噪声
                x = torch.randn(1, seq_length).to(device)
                for t in range(n_steps - 1, -1, -1):
                    t_tensor= torch.full((1,), t, device=device, dtype=torch.long)
                    x = diffusion.p_sample(eps_model, x, t_tensor)

                # 绘制结果
                plt.figure(figsize=(10, 4))
                sin_x = np.linspace(0, 2 * np.pi, seq_length)
                plt.plot(sin_x, x_0[0], '--', label='True sinx')
                plt.plot(sin_x, x[0].cpu().numpy(), label='Generated')
                plt.legend()
                plt.title(f'Epoch {epoch + 1}')
                plt.savefig(f'{dir_name}/sample_epoch_{epoch + 1}.png')
                plt.close()
    print("Training completed!")

    # 保存模型
    torch.save(eps_model.state_dict(), f'{dir_name}/model.pt')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    setup_seed(100)
    main()
