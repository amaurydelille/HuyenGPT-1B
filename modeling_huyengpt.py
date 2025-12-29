import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
from transformers import AutoTokenizer
import torch.utils.data as data
from safetensors.torch import save_file, load_file

import argparse
import logging
import time
import csv
from pathlib import Path

logger = logging.getLogger(__name__)
project_root = Path(__file__).parent
loss_metrics_file = project_root / "loss_metrics.csv"

from pydantic.dataclasses import dataclass

@dataclass
class DatasetConfig:
    dataset_name: str
    input_column: List[str]
    output_column: List[str]


class TextDataset(data.Dataset):
    """
    Dataset for instruction-following with loss masking.
    
    Format: [BOS] <user> prompt <assistant> response [EOS]
    Loss is only computed on the response tokens.
    """
    
    # Special tokens for conversation structure
    USER_TOKEN = "<user>"
    ASSISTANT_TOKEN = "<assistant>"
    
    def __init__(
        self, 
        prompts: List[str],
        responses: List[str],
        tokenizer: AutoTokenizer, 
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add special tokens if not present
        special_tokens = {"additional_special_tokens": [self.USER_TOKEN, self.ASSISTANT_TOKEN]}
        tokenizer.add_special_tokens(special_tokens)
        
        self.examples = []
        for prompt, response in zip(prompts, responses):
            # Handle list responses (take first one)
            if isinstance(response, list):
                response = response[0] if response else ""
            
            # Build the full sequence
            # Format: <user> prompt <assistant> response
            full_text = f"{self.USER_TOKEN} {prompt} {self.ASSISTANT_TOKEN} {response}"
            
            # Tokenize full sequence
            encoded = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True  # Adds BOS/EOS
            )
            
            # Find where the response starts (after <assistant> token)
            # Tokenize just the prompt part to find the boundary
            prompt_part = f"{self.USER_TOKEN} {prompt} {self.ASSISTANT_TOKEN}"
            prompt_encoded = tokenizer(
                prompt_part,
                max_length=max_length,
                truncation=True,
                padding=False,
                return_tensors=None,
                add_special_tokens=True
            )
            prompt_len = len(prompt_encoded["input_ids"])
            
            if len(encoded["input_ids"]) > 1:
                self.examples.append({
                    "input_ids": encoded["input_ids"],
                    "prompt_len": prompt_len  # Where to start computing loss
                })
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> dict:
        example = self.examples[idx]
        input_ids = example["input_ids"]
        prompt_len = example["prompt_len"]
        
        # For causal LM: input = tokens[:-1], target = tokens[1:]
        input_ids_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        labels_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        
        # Mask prompt tokens in labels (set to -100)
        # prompt_len - 1 because labels are shifted by 1
        labels_tensor[:prompt_len - 1] = -100
        
        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
        }


def collate_fn(batch: List[dict], pad_token_id: int) -> dict:
    """Collate function to pad sequences in a batch."""
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Find max length in batch
    max_len = max(len(ids) for ids in input_ids)
    
    # Pad sequences
    padded_input_ids = []
    padded_labels = []
    attention_masks = []
    
    for inp, lbl in zip(input_ids, labels):
        pad_len = max_len - len(inp)
        
        # Pad input_ids (left or right padding - right is more common)
        padded_input_ids.append(F.pad(inp, (0, pad_len), value=pad_token_id))
        
        # Pad labels with -100 (ignored by CrossEntropyLoss)
        padded_labels.append(F.pad(lbl, (0, pad_len), value=-100))
        
        # Attention mask: 1 for real tokens, 0 for padding
        mask = torch.ones(len(inp), dtype=torch.long)
        mask = F.pad(mask, (0, pad_len), value=0)
        attention_masks.append(mask)
    
    return {
        "input_ids": torch.stack(padded_input_ids),        # [B, seq_len]
        "labels": torch.stack(padded_labels),              # [B, seq_len]
        "attention_mask": torch.stack(attention_masks),    # [B, seq_len]
    }

# model variants: "large", "medium", "small", "tiny"
# Use "tiny" for MPS/testing, "small"+ for CUDA with more VRAM
VARIANT = "small"
TOKENIZER_NAME = "google/mt5-small"

# model parameters by variant
MODEL_CONFIGS = {
    "large":  {"d_model": 8192, "d_hidden": 14336, "n_latents": 32, "d_latent": 4096, "n_heads": 32, "n_layers": 32, "n_experts": 8},
    "medium": {"d_model": 4096, "d_hidden": 7168,  "n_latents": 16, "d_latent": 2048, "n_heads": 32, "n_layers": 16, "n_experts": 4},
    "small":  {"d_model": 2048, "d_hidden": 3584,  "n_latents": 8,  "d_latent": 1024, "n_heads": 16, "n_layers": 8,  "n_experts": 2},
    "tiny":   {"d_model": 512,  "d_hidden": 1024,  "n_latents": 4,  "d_latent": 256,  "n_heads": 8,  "n_layers": 4,  "n_experts": 2},
}

config = MODEL_CONFIGS[VARIANT]
D_MODEL = config["d_model"]
D_HIDDEN = config["d_hidden"]
N_LATENTS = config["n_latents"]
D_LATENT = config["d_latent"]
N_HEADS = config["n_heads"]
N_LAYERS = config["n_layers"]
N_EXPERTS = config["n_experts"]
ACTIVE_EXPERTS_PER_TOKEN = min(2, N_EXPERTS)

# Device selection - fall back to CPU if MPS has issues with large models
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available() and VARIANT in ["tiny", "small"]:
    DEVICE = "mps"
else:
    DEVICE = "cpu"  # Safer fallback for larger models

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
PAD_TOKEN_ID = tokenizer.pad_token_id

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

    def _attn(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        causal: bool = False,
        padding_mask: torch.Tensor = None  # [B, seq_len_k] True = masked
    ) -> torch.Tensor:
        # Q: [B, n_heads, seq_len_q, d_h]
        # K: [B, n_heads, seq_len_k, d_h]
        scores = (Q @ K.transpose(-1, -2)) / (self.d_h ** 0.5)  # [B, n_heads, seq_len_q, seq_len_k]
        
        # Causal mask: prevent attending to future tokens
        if causal:
            seq_len_q, seq_len_k = Q.size(-2), K.size(-2)
            causal_mask = torch.triu(
                torch.ones(seq_len_q, seq_len_k, device=Q.device, dtype=torch.bool), 
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Padding mask: prevent attending to padding tokens
        if padding_mask is not None:
            # padding_mask: [B, seq_len_k] -> [B, 1, 1, seq_len_k]
            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        # Handle NaN from all-masked rows (replace with zeros)
        weights = torch.nan_to_num(weights, nan=0.0)
        return weights @ V

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        padding_mask: torch.Tensor = None  # [B, seq_len] True = padding
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # expand latents per batch
        L = self.L.unsqueeze(0).expand(B, self.n_latents, self.d_latent)

        # compression: latents attend to inputs (with causal mask on inputs)
        Q = self._split_heads(self.q_lat(L))
        K = self._split_heads(self.k_in(x))
        K = RoPE.apply_rope(K, cos=cos, sin=sin)       
        V = self._split_heads(self.v_in(x))
        z = self._merge_heads(self._attn(Q, K, V, causal=True, padding_mask=padding_mask))

        # reasoning: latent self-attention (no masks - latents are position-agnostic)
        Ql = self._split_heads(self.q_lat(z))
        Kl = self._split_heads(self.k_lat(z))
        Vl = self._split_heads(self.v_lat(z))
        z2 = self._merge_heads(self._attn(Ql, Kl, Vl, causal=False, padding_mask=None))

        # decompression: tokens attend to latents (no masks on latents)
        Qx = self._split_heads(self.q_in(x))
        Qx = RoPE.apply_rope(Qx, cos=cos, sin=sin)
        Kz = self._split_heads(self.k_lat(z2))
        Vz = self._split_heads(self.v_lat(z2))
        x_latent = self._merge_heads(self._attn(Qx, Kz, Vz, causal=False, padding_mask=None))

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
    def __init__(self, d_model: int, d_hidden: int, n_latents: int, d_latent: int, n_heads: int, n_experts: int, top_k: int) -> None:
        super().__init__()
        
        self.residual_connection_1 = ResidualConnection(d_model=d_model)
        self.mlha = MultiLatentHeadAttention(d_model=d_model, n_latents=n_latents, d_latent=d_latent, n_heads=n_heads)
        self.residual_connection_2 = ResidualConnection(d_model=d_model)
        self.moe = MixtureOfExperts(d_model=d_model, d_hidden=d_hidden, n_experts=n_experts, top_k=top_k)
        self.residual_connection_3 = ResidualConnection(d_model=d_model)
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(
        self, 
        x: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        x = self.residual_connection_1(x, self.mlha(x, cos, sin, padding_mask))
        x = self.residual_connection_2(x, self.moe(x))
        x = self.residual_connection_3(x, self.linear(x))
        return x


class Decoder(nn.Module):
    """Stack of decoder layers (transformer backbone without embedding/LM head)."""
    
    def __init__(
        self, 
        d_model: int, 
        d_hidden: int, 
        n_latents: int, 
        d_latent: int, 
        n_heads: int, 
        n_layers: int, 
        n_experts: int, 
        top_k: int,
        max_seq_length: int = 1024
    ) -> None:
        super().__init__()
        self.d_model = d_model
        head_dim = d_latent // n_heads  # RoPE is applied in latent space
        
        # RoPE cache will be built on first forward (to get correct device)
        self.head_dim = head_dim
        self.max_seq_length = max_seq_length
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                d_model=d_model, 
                d_hidden=d_hidden, 
                n_latents=n_latents, 
                d_latent=d_latent, 
                n_heads=n_heads, 
                n_experts=n_experts, 
                top_k=top_k
            ) for _ in range(n_layers)
        ])
        
        self.final_norm = LayerNorm(d_model)
    
    def _init_rope_cache(self, device: torch.device):
        """Initialize RoPE cache on the correct device."""
        if self.cos is None:
            cos, sin = RoPE.build_rope_cache(self.max_seq_length, self.head_dim, device)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        # Initialize RoPE on correct device
        self._init_rope_cache(x.device)
        
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, padding_mask)
        
        return self.final_norm(x)


class HuyenGPT(nn.Module):
    """
    Complete LLM with embedding, decoder, and language model head.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_hidden: int,
        n_latents: int,
        d_latent: int,
        n_heads: int,
        n_layers: int,
        n_experts: int,
        top_k: int,
        max_seq_length: int = 1024,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.embed_scale = d_model ** 0.5
        
        self.decoder = Decoder(
            d_model=d_model,
            d_hidden=d_hidden,
            n_latents=n_latents,
            d_latent=d_latent,
            n_heads=n_heads,
            n_layers=n_layers,
            n_experts=n_experts,
            top_k=top_k,
            max_seq_length=max_seq_length,
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        if self.pad_token_id is not None:
            self.embedding.weight.data[self.pad_token_id].zero_()
    
    def resize_token_embeddings(self, new_vocab_size: int):
        """Resize embedding and lm_head for new vocabulary size."""
        old_vocab_size = self.vocab_size
        if new_vocab_size == old_vocab_size:
            return
        
        new_embedding = nn.Embedding(new_vocab_size, self.d_model, padding_idx=self.pad_token_id)
        new_embedding.weight.data[:old_vocab_size] = self.embedding.weight.data
        
        nn.init.normal_(new_embedding.weight.data[old_vocab_size:], mean=0.0, std=0.02)
        
        self.embedding = new_embedding
        self.lm_head = nn.Linear(self.d_model, new_vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.vocab_size = new_vocab_size
    
    def forward(
        self,
        input_ids: torch.Tensor,           # [B, seq_len]
        attention_mask: torch.Tensor = None,  # [B, seq_len] 1=real, 0=pad
        labels: torch.Tensor = None,       # [B, seq_len] for loss computation
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            logits: [B, seq_len, vocab_size]
            loss: scalar tensor if labels provided, else None
        """
        B, seq_len = input_ids.shape
        
        # Create padding mask from attention_mask (True = masked/padding)
        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        x = self.embedding(input_ids) * self.embed_scale  # [B, seq_len, d_model]

        hidden = self.decoder(x, padding_mask)  # [B, seq_len, d_model]

        logits = self.lm_head(hidden)  # [B, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss
        
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        eos_token_id: int = None,
    ):
        """
        Autoregressive text generation with streaming.
        
        Yields:
            int: Token ID for each generated token (does NOT include input tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]
            
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_value = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            token_id = next_token.item()
            yield token_id
            
            if eos_token_id is not None and token_id == eos_token_id:
                break
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: str = "cpu",
    ) -> "HuyenGPT":
        """
        Load a pretrained model from a checkpoint file.
        
        Args:
            checkpoint_path: Path to .pth, .bin, or .safetensors file
            device: Device to load model on ("cpu", "cuda", "mps")
            
        Returns:
            HuyenGPT: Loaded model ready for inference
        """
        from safetensors.torch import load_file
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # ... existing code in from_pretrained method ...

        suffix = checkpoint_path.suffix.lower()
        device = torch.device(device)
        
        def get_config_from_d_model(d_model):
            for variant, cfg in MODEL_CONFIGS.items():
                if cfg["d_model"] == d_model:
                    return cfg
            return MODEL_CONFIGS[VARIANT]
        
        if suffix == ".safetensors":
            state_dict = load_file(str(checkpoint_path))
            vocab_size = state_dict["embedding.weight"].shape[0]
            d_model = state_dict["embedding.weight"].shape[1]
            cfg = get_config_from_d_model(d_model)
            model = cls(
                vocab_size=vocab_size,
                d_model=d_model,
                d_hidden=cfg["d_hidden"],
                n_latents=cfg["n_latents"],
                d_latent=cfg["d_latent"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                n_experts=cfg["n_experts"],
                top_k=min(2, cfg["n_experts"]),
            )
            model.load_state_dict(state_dict)
            
        elif suffix in [".pth", ".bin", ".pt"]:
            checkpoint = torch.load(str(checkpoint_path), map_location=device)
            
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                vocab_size = checkpoint.get("vocab_size", state_dict["embedding.weight"].shape[0])
                d_model = checkpoint.get("d_model", state_dict["embedding.weight"].shape[1])
            else:
                state_dict = checkpoint
                vocab_size = state_dict["embedding.weight"].shape[0]
                d_model = state_dict["embedding.weight"].shape[1]
            
            cfg = get_config_from_d_model(d_model)
            model = cls(
                vocab_size=vocab_size,
                d_model=d_model,
                d_hidden=cfg["d_hidden"],
                n_latents=cfg["n_latents"],
                d_latent=cfg["d_latent"],
                n_heads=cfg["n_heads"],
                n_layers=cfg["n_layers"],
                n_experts=cfg["n_experts"],
                top_k=min(2, cfg["n_experts"]),
            )
            model.load_state_dict(state_dict)

# ... existing code ...
        else:
            raise ValueError(f"Unsupported format: {suffix}")
        
        model = model.to(device)
        model.eval()
        return model
    
    def generate_text(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ):
        """
        High-level text generation with streaming.
        
        Args:
            prompt: Input text prompt
            tokenizer: Tokenizer to use
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Yields:
            str: Decoded token strings as they are generated
        """
        device = next(self.parameters()).device
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        
        for token_id in self.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        ):
            yield tokenizer.decode([token_id])
class Trainer:
    """Training loop for HuyenGPT."""
    
    def __init__(
        self,
        model: HuyenGPT,
        train_loader: data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epochs: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_interval: int = 10,
        resume_from: Optional[str] = None,
        use_amp: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_interval = log_interval
        
        self.use_amp = use_amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        
        self.global_step = 0
        self.start_epoch = 0
        self.checkpoint_dir = project_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if resume_from:
            self.load_checkpoint(resume_from)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("global_step", 0)
        
        logger.info(f"Resumed from epoch {self.start_epoch}, global_step {self.global_step}")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            is_accumulation_step = (batch_idx + 1) % self.gradient_accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == num_batches
            
            if is_accumulation_step or is_last_batch:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            if batch_idx % self.log_interval == 0:
                current_loss = loss.item() * self.gradient_accumulation_steps
                logger.info(f"Epoch {epoch+1} | Batch {batch_idx}/{num_batches} | Loss: {current_loss:.4f}")
                
                with open(loss_metrics_file, 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow([epoch + 1, batch_idx, current_loss, self.global_step])
        
        return total_loss / max(1, num_batches)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "vocab_size": self.model.vocab_size,
            "d_model": self.model.d_model,
        }
        
        base = self.checkpoint_dir / f"huyengpt_epoch_{epoch + 1}"
        torch.save(state, str(base.with_suffix(".pth")))
        
        # For safetensors: clone tensors to avoid shared memory issue (tied weights)
        state_dict_cloned = {k: v.clone() for k, v in self.model.state_dict().items()}
        save_file(state_dict_cloned, str(base.with_suffix(".safetensors")))
        
        if is_best:
            best_path = self.checkpoint_dir / "huyengpt_best.pth"
            torch.save(state, str(best_path))
        
        logger.info(f"Saved checkpoint: {base}")

    def train(self) -> None:
        """Full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs (starting from epoch {self.start_epoch + 1})")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_loss = float('inf')
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.epochs):
                epoch_start = time.time()
                logger.info(f"\n{'='*50}")
                logger.info(f"Epoch {epoch + 1}/{self.epochs}")
                logger.info(f"{'='*50}")
                
                avg_loss = self.train_epoch(epoch)
                epoch_time = time.time() - epoch_start
                
                logger.info(f"Epoch {epoch + 1} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
                
                is_best = avg_loss < best_loss
                if is_best:
                    best_loss = avg_loss
                self.save_checkpoint(epoch, is_best=is_best)
            
            total_time = time.time() - start_time
            logger.info(f"\nTraining complete! Total time: {total_time:.1f}s")
            
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            self.save_checkpoint(epoch, is_best=False)
            raise
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            self.save_checkpoint(epoch, is_best=False)
            raise

if __name__ == "__main__":
    from datasets import load_dataset
    from functools import partial
    
    parser = argparse.ArgumentParser(description="Train HuyenGPT")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (.pth)")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(project_root / "training.log")
        ]
    )

    # ========== Configuration ==========
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    MAX_SEQ_LENGTH = 512
    GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4 * 4 = 16
    
    # ========== Load Dataset ==========
    logger.info("Loading dataset...")
    hf_dataset = load_dataset(
        "5CD-AI/Vietnamese-argilla-OpenHermesPreferences-66k-gg-translated", 
        split="train"
    )
    df = hf_dataset.to_pandas()
    
    dataset_config = DatasetConfig(
        dataset_name="Vietnamese-argilla-OpenHermesPreferences-66k-gg-translated",
        input_column=["prompt_en"],
        output_column=["candidates_completions_vi"]
    )

    prompts = df[dataset_config.input_column[0]].tolist()
    responses = df[dataset_config.output_column[0]].tolist()
    
    MAX_SAMPLES = None # Set to None for full dataset
    if MAX_SAMPLES:
        prompts = prompts[:MAX_SAMPLES]
        responses = responses[:MAX_SAMPLES]
        logger.info(f"Using {MAX_SAMPLES} samples for training")
    
    # ========== Create Dataset ==========
    logger.info("Creating dataset...")
    train_dataset = TextDataset(prompts, responses, tokenizer, max_length=MAX_SEQ_LENGTH)
    logger.info(f"Dataset size: {len(train_dataset)} examples")
    
    # ========== Create DataLoader ==========
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    
    # ========== Create Model ==========
    logger.info("Creating model...")
    model = HuyenGPT(
        vocab_size=len(tokenizer),
        d_model=D_MODEL,
        d_hidden=D_HIDDEN,
        n_latents=N_LATENTS,
        d_latent=D_LATENT,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_experts=N_EXPERTS,
        top_k=ACTIVE_EXPERTS_PER_TOKEN,
        max_seq_length=MAX_SEQ_LENGTH,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    device = torch.device(DEVICE)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    
    # ========== Train ==========
    logger.info("\nStarting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=1.0,
        log_interval=10,
        resume_from=args.resume,
    )
    
    trainer.train()
    
    # second phase
    dataset_config = DatasetConfig(
        dataset_name="Vietnamese-argilla-OpenHermesPreferences-66k-gg-translated",
        input_column=["prompt_vi"],
        output_column=["candidates_completions_vi"]
    )

    prompts = df[dataset_config.input_column[0]].tolist()
    responses = df[dataset_config.output_column[0]].tolist()
    
    MAX_SAMPLES = None # Set to None for full dataset
    if MAX_SAMPLES:
        prompts = prompts[:MAX_SAMPLES]
        responses = responses[:MAX_SAMPLES]
        logger.info(f"Using {MAX_SAMPLES} samples for training")
    
    # ========== Create Dataset ==========
    logger.info("Creating dataset...")
    train_dataset = TextDataset(prompts, responses, tokenizer, max_length=MAX_SEQ_LENGTH)
    logger.info(f"Dataset size: {len(train_dataset)} examples")
    
    # ========== Create DataLoader ==========
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_grad_norm=1.0,
        log_interval=10,
        resume_from=args.resume,
    )
    
    trainer.train()
    
    