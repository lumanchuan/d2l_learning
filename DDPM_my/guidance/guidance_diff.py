import random

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from NoisePredictNet import NoisePredictor
import numpy as np
import os

# 扩散模型
class GuidanceDenoiseDiffusion:
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

    def p_sample(self, eps_model, x_t, t, vocab):
        """去噪步骤"""
        alpha_t = self.alpha[t][:, None]
        alpha_bar_t = self.alpha_bar[t][:, None]
        eps_theta = eps_model(x_t, vocab, t)
        mean = (1. / torch.sqrt(alpha_t)) * (x_t - (1. - alpha_t) / torch.sqrt(1. - alpha_bar_t) * eps_theta)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(self.beta[t])[:, None]
            return mean + sigma_t * noise
        else:
            return mean

    def loss(self, eps_model, x_vocab):
        x_0 = torch.tensor([item[0] for item in x_vocab], dtype=torch.float32, device=self.device)
        vocabs = [item[1] for item in x_vocab]

        batch_size = x_0.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(x_0)
        x_t, target = self.q_sample(x_0, t, noise)
        eps_theta = eps_model(x_t, vocabs, t)

        return torch.nn.functional.mse_loss(eps_theta, target)

def generate_data(batch_size, seq_length):
    """生成正弦波训练数据"""
    x = np.linspace(0, 2 * np.pi, seq_length)
    data = []
    for _ in range(batch_size // 2):
        y = np.sin(x)
        data.append((y, 'sin'))
        y = np.cos(x)
        data.append((y, 'cos'))
    return data


def main():
    # 设置参数
    len_seq = 64
    batch_size = 32
    n_steps = 1000
    n_epochs = 1_000_000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_name = f'results_step{n_steps}_epoch{n_epochs}'

    # 创建保存结果的文件夹
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # 初始化模型
    eps_model = NoisePredictor(len_seq, 2).to(device)
    diffusion = GuidanceDenoiseDiffusion(n_steps=n_steps, device=device)
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(n_epochs):
        # 生成训练数据
        x_vocab = generate_data(batch_size, len_seq)
        # 训练步骤
        optimizer.zero_grad()
        loss = diffusion.loss(eps_model, x_vocab)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100_000 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
            # 生成样本
            with torch.no_grad():
                # 去噪起点为纯噪声生成sin
                x_sin = torch.randn(1, len_seq).to(device)
                for t in range(n_steps - 1, -1, -1):
                    t = torch.full((1,), t, device=device)
                    x_sin = diffusion.p_sample(eps_model, x_sin, t, ['sin'])

                # 绘制结果
                plt.figure(figsize=(10, 4))
                sin_x = np.linspace(0, 2 * np.pi, len_seq)
                plt.plot(sin_x, np.sin(sin_x), '--', label='True sinx')
                plt.plot(sin_x, x_sin[0].cpu().numpy(), label='Generated')
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
