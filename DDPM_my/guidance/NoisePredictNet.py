import torch
import torch.nn as nn



class NoisePredictor(torch.nn.Module):
    def __init__(self, len_seq, vocab_size, drop_out=0.15, hidden_dim=32):
        super().__init__()
        self.len_seq = len_seq
        self.hidden_dim = hidden_dim
        self.drop_out = drop_out
        self.norm = nn.LayerNorm(self.len_seq * 2)

        # 词嵌入
        self.vocab_size = vocab_size
        self.vocab_to_idx = {"cos": 0, "sin": 1}

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.layer = nn.Sequential(
            nn.Linear(self.len_seq * 2, self.len_seq * 2),
            nn.Dropout(self.drop_out),
            nn.SiLU(),
            nn.Linear(self.len_seq * 2, self.len_seq * 2),
            nn.Dropout(self.drop_out),
            nn.SiLU(),
            nn.Linear(self.len_seq * 2, self.len_seq * 2)
        )

        self.net = nn.Sequential(
            nn.Linear(self.len_seq * 2, self.len_seq * 2),
            nn.Dropout(self.drop_out),
            nn.SiLU(),
            nn.Linear(self.len_seq * 2, self.len_seq * 2),
            nn.Dropout(self.drop_out),
            nn.SiLU(),
            nn.Linear(self.len_seq * 2, self.len_seq)
        )

    def vocab_embedding(self, vocabs):
        embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        indices = [self.vocab_to_idx[word] for word in vocabs]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        return embedding(indices_tensor)

    def forward(self, x, vocab, t):
        vocab_emb = self.vocab_embedding(vocab).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        t = t.unsqueeze(-1).float()
        time_emb = self.time_mlp(t)
        x = torch.cat([x, vocab_emb, time_emb], dim=1)
        x = self.layer(x) + x
        x = self.norm(x)
        x = self.layer(x) + x
        x = self.norm(x)
        # 预测噪声
        noise_pred = self.net(x)  # [batch_size, seq_length]
        return noise_pred


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, len_seq = 1, 64
    model = NoisePredictor(len_seq,  vocab_size=2)
    x = torch.randn(batch_size, len_seq)
    t = torch.randint(0, 100, (batch_size,))
    output = model(x, ['cos'], t)
    print("Output shape:", output.shape)
