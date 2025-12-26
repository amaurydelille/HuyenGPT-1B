import torch
import torch.nn as nn
import torch.functional as F
from typing import Tuple

N_EXPERTS=8
ACTIVE_EXPERTS_PER_TOKEN=2

class RoPE:
    @staticmethod
    def build_rope_cache(max_seq_length: int, head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        half_dim = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device) / half_dim))
        positions = torch.arange(max_seq_length, device=device)
        angles = torch.einsum("i,j->ij", positions, inv_freq)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, h, s, d = x.shape
        x = x.view(b, h, s, d // 2, 2)
        x1, x2 = x[..., 0], x[..., 1]
        cos = cos[:s].unsqueeze(0).unsqueeze(0)
        sin = sin[:s].unsqueeze(0).unsqueeze(0)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).reshape(b, h, s, d)

class LoadBalancingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self) -> torch.Tensor:
        pass

class MultiLatentHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_latents: int, d_latent: int, n_heads: int) -> None:
        assert d_latent % n_heads == 0, "d_latent should be divisible by n_heads."
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.n_latents = n_latents
        self.n_heads = n_heads
        self.d_h = d_latent // n_heads

        # latent set
        self.L = nn.Parameter(torch.randn(n_latents, d_latent))

        # projections
        self.q_lat = nn.Linear(d_latent, d_latent, bias=False)    # queries from latents
        self.k_in  = nn.Linear(d_model,  d_latent, bias=False)    # keys from inputs
        self.v_in  = nn.Linear(d_model,  d_latent, bias=False)    # values from inputs

        self.q_in  = nn.Linear(d_model,  d_latent, bias=False)    # queries from inputs
        self.k_lat = nn.Linear(d_latent, d_latent, bias=False)    # keys from latents
        self.v_lat = nn.Linear(d_latent, d_latent, bias=False)    # values from latents

        self.out_proj = nn.Linear(d_latent, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_latent] -> [B, n_heads, T, d_h]
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_h).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_heads, T, d_h] -> [B, T, d_latent]
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_h)

    def _attn(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        scores = (Q @ K.transpose(-1, -2)) / (self.d_h ** 0.5)
        weights = F.softmax(scores, dim=-1)
        return weights @ V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        # expand latents per batch
        L = self.L.unsqueeze(0).expand(B, self.n_latents, self.d_latent)

        # compression: latents attend to inputs
        Q = self._split_heads(self.q_lat(L))
        K = self._split_heads(self.k_in(x))
        V = self._split_heads(self.v_in(x))
        z = self._merge_heads(self._attn(Q, K, V))  # [B, n_latents, d_latent]

        # reasoning: latent self-attention
        Ql = self._split_heads(self.q_lat(z))
        Kl = self._split_heads(self.k_lat(z))
        Vl = self._split_heads(self.v_lat(z))
        z2 = self._merge_heads(self._attn(Ql, Kl, Vl))  # [B, n_latents, d_latent]

        # decompression: tokens attend to latents
        Qx = self._split_heads(self.q_in(x))
        Kz = self._split_heads(self.k_lat(z2))
        Vz = self._split_heads(self.v_lat(z2))
        x_latent = self._merge_heads(self._attn(Qx, Kz, Vz))  # [B, N, d_latent]

        return self.out_proj(x_latent)
    
class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network
    SwiGLU introduces additional non-linearity, enables conditional computation,
    feature-wise modulation without adding more parameters.
    So you get higher expressivity per parameter thus better scaling laws.
    """

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden

        self.W = nn.Parameter(torch.randn(d_model, d_hidden))
        self.V = nn.Parameter(torch.randn(d_model, d_hidden))

    def __swish(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x @ self.W
        b = x @ self.V

        return a * self.__swish(b)
    
class LayerNorm(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.gamma * (x - mean) / (std + 1e-6) + self.beta
    
class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer with top-k routing and SwiGLU feed-forward experts."""

    def __init__(self, d_module: int, d_hidden: int, n_experts: int, top_k: int) -> None:
        assert top_k <= n_experts, "top_k cannot be greater than n_experts."
        super().__init__()

        self.experts = nn.ModuleList([
            SwiGLU(d_model=d_model, d_hidden=d_hidden) for _ in range(n_experts)
        ])
        self.router = nn.Linear(in_features=d_module, out_features=n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MoE layer.

        Notes about MoE:
        The routing is done at token-level, it means that each token in the sequence
        can be router to different k experts. Also, each of this token will be
        split into k different versions, with each of them being processed by one expert.

        After processing, the outputs from different experts are aggregated back
        for each token using a weighted sum based on the routing probabilities.
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D) # flatten tokens

        router_logits = self.router(x) 
        top_k_logits, top_k_ids = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1) # final probabilities after softmax

        output = torch.zeros(D)

        for expert_id, expert in enumerate(self.experts):
            mask = (top_k_ids == expert_id) # mask to find tokens assigned to this expert
            token_idx = mask.any(dim=-1).nonzero(as_tuple=True)[0]

            if len(token_idx) == 0: # no tokens for this expert
                continue

            token_positions = mask[token_idx].nonzero(as_tuple=False)
            rows = token_positions[:, 0]
            cols = token_positions[:, 1]
            weights = top_k_probs[token_idx, cols].unsqueeze(-1)

            x_input = x_flat[rows]
            y = expert(x_input)

            output[rows] += y * weights

        return output.view(B, T, D)
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_latents: int, d_latent: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        pass

class Decoder(nn.Module):
    def __init__(self, d_model: int, n_latents: int, d_latent: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        head_dim = d_model // n_heads
        cos, sin = RoPE.build_rope_cache(max_seq_length=1024, head_dim=head_dim, device=torch.device("cuda"))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, n_latents=n_latents, d_latent=d_latent, n_heads=n_heads, n_layers=n_layers) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = RoPE.apply_rope(x, self.cos, self.sin)
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    d_model = 4
    X = torch.randn(1, 3, d_model)