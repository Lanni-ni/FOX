import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from fox_attention import ForgetAttention  

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=256, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.attn = ForgetAttention(hidden_size=hidden_size, num_attention_heads=num_heads)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        emb = self.embed(x).transpose(0, 1)  # [seq_len, batch, hidden]
        out, _ = self.attn(emb)
        out = out.transpose(0, 1)  # [batch, seq_len, hidden]
        return self.linear(out)


if __name__ == "__main__":
    seq_len = 4
    vocab_size = 100
    batch_size = 2

    X = torch.randint(0, vocab_size, (20, seq_len))
    Y = X.clone()

    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MiniTransformer(vocab_size=vocab_size, hidden_size=256, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for step, (x, y) in enumerate(loader):
        logits = model(x)
        loss = criterion(logits.view(-1, vocab_size), y.view(-1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        print(f"Step {step} | Loss: {loss.item():.4f}")

        if step == 5:
            break  
