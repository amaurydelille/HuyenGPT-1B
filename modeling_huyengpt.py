import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from transformers import AutoTokenizer

# model variants
VARIANT="small"
TOKENIZER_NAME="google/mt5-small"

# model parameters
D_MODEL=8192 if VARIANT=="large" else 4096 if VARIANT=="medium" else 2048 if VARIANT=="small" else 1024
D_HIDDEN=14336 if VARIANT=="large" else 7168 if VARIANT=="medium" else 3584 if VARIANT=="small" else 1792
N_LATENTS=32 if VARIANT=="large" else 16 if VARIANT=="medium" else 8 if VARIANT=="small" else 4
D_LATENT=4096 if VARIANT=="large" else 2048 if VARIANT=="medium" else 1024 if VARIANT=="small" else 512
N_HEADS=32
N_LAYERS=32 if VARIANT=="large" else 16 if VARIANT=="medium" else 8 if VARIANT=="small" else 4
N_EXPERTS=8 if VARIANT=="large" else 4 if VARIANT=="medium" else 2 if VARIANT=="small" else 1
ACTIVE_EXPERTS_PER_TOKEN=2
DEVICE="cuda" if torch.cuda.is_available() else "mps"



tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

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
        self.L = nn.Parameter(torch.randn(n_latents, d_latent) * (d_latent ** -0.5))

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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        # expand latents per batch
        L = self.L.unsqueeze(0).expand(B, self.n_latents, self.d_latent)

        # compression: latents attend to inputs
        Q = self._split_heads(self.q_lat(L))
        K = self._split_heads(self.k_in(x))
        K = RoPE.apply_rope(K, cos=cos, sin=sin)       
        V = self._split_heads(self.v_in(x))
        z = self._merge_heads(self._attn(Q, K, V))  # [B, n_latents, d_latent]

        # reasoning: latent self-attention
        Ql = self._split_heads(self.q_lat(z))
        Kl = self._split_heads(self.k_lat(z))
        Vl = self._split_heads(self.v_lat(z))
        z2 = self._merge_heads(self._attn(Ql, Kl, Vl))  # [B, n_latents, d_latent]

        # decompression: tokens attend to latents
        Qx = self._split_heads(self.q_in(x))
        Qx = RoPE.apply_rope(Qx, cos=cos, sin=sin)
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

        # Xavier initialization: scale by 1/sqrt(fan_in)
        self.W = nn.Parameter(torch.randn(d_model, d_hidden) * (d_model ** -0.5))
        self.V = nn.Parameter(torch.randn(d_model, d_hidden) * (d_model ** -0.5))
        self.W_out = nn.Parameter(torch.randn(d_hidden, d_model) * (d_hidden ** -0.5))

    def __swish(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.sigmoid(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x @ self.W
        b = x @ self.V
        hidden = a * self.__swish(b)
        return hidden @ self.W_out  # project back to d_model

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x) + y
    
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

    def __init__(self, d_model: int, d_hidden: int, n_experts: int, top_k: int) -> None:
        assert top_k <= n_experts, "top_k cannot be greater than n_experts."
        super().__init__()

        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            SwiGLU(d_model=d_model, d_hidden=d_hidden) for _ in range(n_experts)
        ])
        self.router = nn.Linear(in_features=d_model, out_features=n_experts)

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
        x_flat = x.view(N, D)  # flatten tokens

        router_logits = self.router(x_flat)  # [N, n_experts]
        top_k_logits, top_k_ids = torch.topk(router_logits, k=self.top_k, dim=-1)  # [N, top_k]
        top_k_probs = F.softmax(top_k_logits, dim=-1)  # [N, top_k]

        output = torch.zeros(N, D, device=x.device, dtype=x.dtype)

        for expert_id, expert in enumerate(self.experts):
            mask = (top_k_ids == expert_id)  # [N, top_k] - which slots have this expert
            token_idx, slot_idx = mask.nonzero(as_tuple=True)  # tokens and their slots assigned to this expert

            if len(token_idx) == 0:
                continue

            weights = top_k_probs[token_idx, slot_idx].unsqueeze(-1)  # [num_tokens, 1]
            x_input = x_flat[token_idx]  # [num_tokens, D]
            y = expert(x_input)  # [num_tokens, D]

            output.index_add_(0, token_idx, y * weights)

        return output.view(B, T, D)
        
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_latents: int, d_latent: int, n_heads: int, n_layers: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        
        self.residual_connection_1 = ResidualConnection(d_model=d_model)
        self.mlha = MultiLatentHeadAttention(d_model=d_model, n_latents=n_latents, d_latent=d_latent, n_heads=n_heads)
        self.residual_connection_2 = ResidualConnection(d_model=d_model)
        self.moe = MixtureOfExperts(d_model=d_model, d_hidden=d_hidden, n_experts=n_experts, top_k=top_k)
        self.residual_connection_3 = ResidualConnection(d_model=d_model)
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = self.residual_connection_1(x, self.mlha(x, cos, sin))
        x = self.residual_connection_2(x, self.moe(x))
        x = self.residual_connection_3(x, self.linear(x))
        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, n_latents: int, d_latent: int, n_heads: int, n_layers: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        head_dim = d_latent // n_heads  # RoPE is applied in latent space
        cos, sin = RoPE.build_rope_cache(max_seq_length=1024, head_dim=head_dim, device=torch.device(DEVICE))
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_hidden=d_hidden, n_latents=n_latents, d_latent=d_latent, n_heads=n_heads, n_layers=n_layers, n_experts=n_experts, top_k=top_k) for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, self.cos, self.sin)
        return x


if __name__ == "__main__":
    X = torch.randn(1, 3, D_MODEL, device=torch.device(DEVICE))
    decoder = Decoder(d_model=D_MODEL, d_hidden=D_HIDDEN, n_latents=N_LATENTS, d_latent=D_LATENT, n_heads=N_HEADS, n_layers=N_LAYERS, n_experts=N_EXPERTS, top_k=ACTIVE_EXPERTS_PER_TOKEN).to(DEVICE)
    print(sum(p.numel() for p in decoder.parameters()))
    print(decoder(X).sum())