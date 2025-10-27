import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from dataclasses import dataclass
from transformer_lens import HookedTransformer
from jaxtyping import Float, Int
from transformer_lens.utils import gelu_new
import einops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

reference_text = "Today we are going to implement a Transformer from scratch!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
logits, cache = reference_gpt2.run_with_cache(tokens)

@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

cfg = Config()


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:
        # implement your solution here
        pass

class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        #implement your solution here
        pass




class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        #implement your solution here
        pass


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(torch.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(torch.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE",torch.tensor(-1e5, dtype=torch.float32, device=device))

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        ones = torch.ones_like(attn_scores)
        mask = torch.triu(ones, diagonal=1)
        mask = mask.to(torch.bool)
        return attn_scores.masked_fill(mask, self.IGNORE)

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Linear Mapping: compute matrices Q, K, and V
        q = einops.einsum(normalized_resid_pre, self.W_Q, '[fill in pattern here]') + self.b_Q
        k = einops.einsum(normalized_resid_pre, self.W_K, '[fill in pattern here]') + self.b_K
        v = einops.einsum(normalized_resid_pre, self.W_V, '[fill in pattern here]') + self.b_V

        # dot product to compute attention scores
        a = einops.einsum(q, k, '[fill in pattern here]')

        # re-scale
        a = a / (self.cfg.d_head ** 0.5)

        # apply causal mask
        a = self.apply_causal_mask(a)

        # apply softmax
        a = a.softmax(dim=-1)

        # get get weighted sum of values
        z = einops.einsum(v, a, '[fill in pattern here]')

        # sum over different heads
        attn_out = einops.einsum(z, self.W_O, '[fill in pattern here]') + self.b_O

        return attn_out


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(torch.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(torch.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
      pass


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        #implement your solution here
        pass


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(torch.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        #implement your solution here
        pass

class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        #implement your solution here
        pass

def greedy_decode(model, start_tokens, max_new_tokens):
    """
    Generates text using greedy decoding from an initial sequence of tokens.

    Parameters:
    - model: The pre-trained DemoTransformer model.
    - start_tokens: A list of initial tokens (integers) to start the text generation.
    - max_new_tokens: The maximum number of new tokens to generate.

    Returns:
    - A list of generated token IDs, including the start_tokens.
    """
    with torch.no_grad():  # Disable gradient calculation for inference
        generated = start_tokens # Shape: [1, seq_len]


        pass
        # implement your solution here



    return generated

if __name__ == "__main__":
    # Load reference model only when running this file directly
    reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
    
    reference_text = "Today we are going to implement a Transformer from scratch!"
    tokens = reference_gpt2.to_tokens(reference_text).to(device)
    logits, cache = reference_gpt2.run_with_cache(tokens)
    
    demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
    demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

    start_sequence = "Today I was walking home, when suddenly"
    start_tokens = reference_gpt2.to_tokens(start_sequence, prepend_bos=True)
    max_new_tokens = 20

    generated_tokens = greedy_decode(demo_gpt2, start_tokens, max_new_tokens)
    # generated_text = reference_gpt2.to_string(generated_tokens[1:])
    # generated_text = reference_gpt2.to_string(generated_tokens[0]) 
    generated_text = reference_gpt2.to_string(generated_tokens[0, 1:])
    print("Generated Text:", generated_text)

    reference_generation = reference_gpt2.generate(start_sequence, max_new_tokens=max_new_tokens, stop_at_eos=False, do_sample=False)
    assert reference_generation == generated_text, f'{reference_generation} vs {generated_text}'
    print('The generations match!')
