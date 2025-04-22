import torch
import torch.nn as nn
import torch.nn.functional as F

class ForgetAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, dropout_p=0.0, use_cache=False, kernel_fn=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout_p = dropout_p
        self.use_cache = use_cache

        assert hidden_size % num_attention_heads == 0

        #q.k,v
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)

        self.forget_gate = nn.Linear(hidden_size, num_attention_heads)
        self.out_proj = nn.Linear(hidden_size, hidden_size)


        #kernel（linear attention)
        if kernel_fn is None:
            self.kernel_fn = lambda x: F.elu(x) + 1
        else:
            self.kernel_fn = kernel_fn

    def forward(self, hidden_states,  attention_mask=None, layer_past=None):

        seq_len, batch_size, _ = hidden_states.shape
        def shape(x):
            x = x.view(seq_len, batch_size, self.num_heads, self.head_dim)
            return x

        q = shape(self.q(hidden_states))
        k = shape(self.k(hidden_states))
        v = shape(self.v(hidden_states))

        phi_q = self.kernel_fn(q)
        phi_k = self.kernel_fn(k)

        f = torch.sigmoid(self.forget_gate(hidden_states)).view(seq_len, batch_size, self.num_heads)
        # f [seq_length, batch_size]
        output = torch.zeros_like(v)

        #recurrent \phi(q_t)
        for t in range(seq_len):
            phi_qt = phi_q[t]  # [batch, num_heads, head_dim]
            num = torch.zeros(batch_size, self.num_heads, self.head_dim, device=hidden_states.device)
            denom = torch.zeros(batch_size, self.num_heads, 1, device=hidden_states.device)

            for j in range(t + 1):
                if t == j:
                    F_tj = torch.ones(batch_size, self.num_heads, device=hidden_states.device)
                else:
                    F_tj = f[j + 1:t + 1].prod(dim=0)  # [batch, num_heads]

                dot = torch.sum(phi_qt * phi_k[j], dim=-1)  # [batch, num_heads]
                weight = F_tj * dot  # [batch, num_heads]
                num += weight.unsqueeze(-1) * v[j]  # [batch, num_heads, head_dim]
                denom += weight.unsqueeze(-1)

            output[t] = num / (denom + 1e-6)

        output = output.view(seq_len, batch_size, self.hidden_size)
        output = self.out_proj(output)


        return output, torch.zeros_like(output), f






