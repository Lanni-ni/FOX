import torch
import torch.nn as nn
from fox_attention import ForgetAttention

# 初始化
hidden_size = 256
num_heads = 4
seq_len = 4
batch_size = 2

model = ForgetAttention(hidden_size, num_attention_heads=num_heads)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# dummy 输入 + target
x = torch.randn(seq_len, batch_size, hidden_size)
target = torch.randn_like(x)

# toy training loop
for step in range(20):
    optimizer.zero_grad()
    output, _, forget_values = model(x)
    loss = ((output - target) ** 2).mean()
    loss.backward()
    optimizer.step()
    print(f"Step {step} | Loss: {loss.item():.4f}")

import torch
from fox_attention import ForgetAttention  # 你定义的模块
import matplotlib.pyplot as plt

# 构造输入
seq_len = 20
batch_size = 2
hidden_size = 256
input_tensor = torch.randn(seq_len, batch_size, hidden_size)

# 初始化模型
model = ForgetAttention(hidden_size=256, num_attention_heads=4)

# 获取输出和 forget gate
output, _, forget_values = model(x)

# 可视化 forget gate
f_np = forget_values.detach().cpu().numpy()
seq_len, batch_size, num_heads = f_np.shape
f_2d = f_np.reshape(seq_len, -1).T

plt.imshow(f_2d, cmap='Blues', aspect='auto')
plt.colorbar(label='Forget Gate Value (sigmoid)')
plt.xlabel('Token Position')
plt.ylabel('Head × Batch')
plt.title('Forget Gate Visualization')
plt.tight_layout()
plt.show()