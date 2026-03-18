"""
Janus Model Rotation Utilities for MQuant Framework

Janus is a multimodal model with a LlamaForCausalLM backbone.
Architecture:
- model.vision_model: Vision encoder (CLIP-based)
- model.aligner: MlpProjector that projects vision features to language model space
  - aligner.layers[0]: Linear(1024 -> 2048)
  - aligner.layers[1]: GELU
  - aligner.layers[2]: Linear(2048 -> 2048) - output to LLM
- model.language_model: LlamaForCausalLM (our quantization target)
  - language_model.model.embed_tokens: Embedding layer
  - language_model.model.layers[i]: Transformer decoder layers
  - language_model.model.norm: Output norm
  - language_model.lm_head: LM head

This module provides rotation utilities specifically for Janus's Llama-based language model.

Key approach (following qwen2vl pattern):
- Rotate text embeddings AND aligner output layer
- This ensures both text and vision features enter the LLM in rotated space
"""

import tqdm
import torch
import typing
from fake_quant import module_util
from fake_quant import utils
from fake_quant.hadamard_utils import (
    random_hadamard_matrix,
    apply_exact_had_to_linear,
    is_pow2,
)


def fuse_ln_linear(
    layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]
) -> None:
    """
    Fuse the linear operations in LayerNorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias") and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(
                    torch.zeros(linear.out_features, dtype=torch.float64).to(W_)
                )
            linear.bias.data = linear.bias.data.double() + torch.matmul(
                W_, layernorm.bias.double()
            )
            linear.bias.data = linear.bias.data.to(linear_dtype)

    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias") and layernorm.bias is not None:
        layernorm.bias.data = torch.zeros_like(layernorm.bias.data)


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")


def fuse_janus_layer_norms(model, args):
    """
    Fuse layer norms for Janus model's language model backbone (LlamaForCausalLM).

    Args:
        model: The VLMEvalKit model wrapper containing model.model as the Janus model
        args: Arguments containing fusion flags
    """
    print("Fusing Janus layer norms...")

    # Access the language model
    language_model = model.model.language_model

    if not args.no_fuse_llm:
        # Fuse layer norms for each transformer layer
        for layer in language_model.model.layers:
            # Fuse input_layernorm into q_proj, k_proj, v_proj
            fuse_ln_linear(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]
            )
            # Fuse post_attention_layernorm into gate_proj, up_proj
            fuse_ln_linear(
                layer.post_attention_layernorm,
                [layer.mlp.gate_proj, layer.mlp.up_proj]
            )

        # Fuse final layer norm into lm_head
        fuse_ln_linear(language_model.model.norm, [language_model.lm_head])

        # Replace layer norms with RMS norm (no learned params)
        for layer in language_model.model.layers:
            hidden_size = language_model.config.hidden_size
            eps = language_model.config.rms_norm_eps if hasattr(language_model.config, 'rms_norm_eps') else 1e-6
            layer.input_layernorm = module_util.RMSN(hidden_size, eps=eps)
            layer.post_attention_layernorm = module_util.RMSN(hidden_size, eps=eps)

        # Replace final norm
        language_model.model.norm = module_util.RMSN(
            language_model.config.hidden_size,
            eps=language_model.config.rms_norm_eps if hasattr(language_model.config, 'rms_norm_eps') else 1e-6
        )

    print("Janus layer norm fusion complete.")


def rotate_janus_embeddings(model, Q) -> None:
    """
    Rotate text embeddings AND aligner output layer.

    This ensures both text and vision features enter the LLM in rotated space:
    - Text embeddings: W_new = W @ Q (input rotation)
    - Aligner output (layers[2]): W_new = Q.T @ W, b_new = b @ Q (output rotation)

    Similar to qwen2vl_rotation.py's rotate_qwen2vl_embeddings.
    """
    language_model = model.model.language_model
    janus_model = model.model  # The full Janus model

    # Rotate text embeddings: W @ Q
    Q_emb = Q.to(language_model.model.embed_tokens.weight.device)
    dtype = language_model.model.embed_tokens.weight.data.dtype
    W_ = language_model.model.embed_tokens.weight.data.to(dtype=torch.float64)
    language_model.model.embed_tokens.weight.data = torch.matmul(W_, Q_emb).to(dtype=dtype)
    print("  ✓ Text embeddings rotated")

    # Rotate aligner output layer (layers[2]): Q.T @ W for output rotation
    # The aligner.layers[2] is the last linear layer that outputs to LLM hidden space
    aligner_out = janus_model.aligner.layers[2]
    Q_aligner = Q.to(aligner_out.weight.device)
    dtype = aligner_out.weight.data.dtype
    W_ = aligner_out.weight.data.to(dtype=torch.float64)
    aligner_out.weight.data = torch.matmul(Q_aligner.T, W_).to(dtype=dtype)
    if aligner_out.bias is not None:
        b = aligner_out.bias.data.to(dtype=torch.float64)
        aligner_out.bias.data = torch.matmul(b, Q_aligner).to(dtype=dtype)
    print("  ✓ Aligner output layer rotated")


def rotate_attention_inputs(layer, Q) -> None:
    """
    Rotate the WQ, WK and WV matrices of the self-attention layer.
    For input rotation: W_new = W @ Q
    """
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(dtype=torch.float64)
        Q_ = Q.to(W.weight.device, torch.float64)
        W.weight.data = torch.matmul(W_, Q_).to(dtype=dtype)


def rotate_attention_output(layer, Q) -> None:
    """
    Rotate output matrix of the self-attention layer.
    For output rotation: W_new = Q.T @ W
    """
    W = layer.self_attn.o_proj

    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(dtype=torch.float64)
    Q_ = Q.to(W.weight.device, torch.float64)
    W.weight.data = torch.matmul(Q_.T, W_).to(dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(dtype=torch.float64)
        W.bias.data = torch.matmul(Q_.T, b).to(dtype=dtype)


def rotate_mlp_input(layer, Q) -> None:
    """
    Rotate the MLP input weights (gate_proj, up_proj).
    For input rotation: W_new = W @ Q
    """
    for W in [layer.mlp.gate_proj, layer.mlp.up_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.data.to(dtype=torch.float64)
        Q_ = Q.to(W.weight.device, torch.float64)
        W.weight.data = torch.matmul(W_, Q_).to(dtype=dtype)


def rotate_mlp_output(layer, Q, online_hadamard=False):
    """
    Rotate the MLP output weights (down_proj).
    For output rotation: W_new = Q.T @ W
    Then apply Hadamard transform if online_hadamard is True.
    """
    out_layer = layer.mlp.down_proj

    dtype = out_layer.weight.data.dtype
    W_ = out_layer.weight.data.to(dtype=torch.float64)
    Q_ = Q.to(out_layer.weight.device, torch.float64)
    out_layer.weight.data = torch.matmul(Q_.T, W_).to(dtype=dtype)

    if online_hadamard:
        # Apply exact (inverse) hadamard on the weights of mlp output
        # had_dim=-1 means full dimension Hadamard
        apply_exact_had_to_linear(out_layer, had_dim=-1, output=False)

    if out_layer.bias is not None:
        b = out_layer.bias.data.to(dtype=torch.float64)
        out_layer.bias.data = torch.matmul(Q_.T, b).to(dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim):
    """
    Apply Hadamard rotation to v_proj output and o_proj input.
    Following qwen2vl: both use head_dim for Hadamard dimension.
    """
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
    apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)


def rotate_head(model, Q: torch.Tensor) -> None:
    """
    Rotate the lm_head of Janus language model.
    For input rotation (lm_head takes rotated hidden state): W_new = W @ Q
    """
    language_model = model.model.language_model

    dtype = language_model.lm_head.weight.data.dtype
    W_ = language_model.lm_head.weight.data.to(dtype=torch.float64)
    Q_ = Q.to(W_.device, torch.float64)
    language_model.lm_head.weight.data = torch.matmul(W_, Q_).to(dtype=dtype)


@torch.no_grad()
def rotate_janus_model(model, args):
    """
    Apply rotation to Janus model's language model backbone.

    Key steps (following qwen2vl pattern):
    1. Pad intermediate size if needed for Hadamard
    2. Generate rotation matrix Q
    3. Rotate embeddings AND aligner output (both text and vision paths)
    4. Rotate lm_head
    5. Rotate transformer layer weights
    6. Apply Hadamard transforms to v_proj/o_proj

    Args:
        model: The VLMEvalKit model wrapper
        args: Arguments containing rotation flags
    """
    print("Rotating Janus model...")

    language_model = model.model.language_model

    if args.rotate_llm:
        # Check if we need to pad intermediate size for Hadamard
        if args.online_llm_hadamard:
            language_model.config.need_pad = False
            from fake_quant.hadamard_utils import auto_pad_size

            new_intermediate_size = auto_pad_size(language_model.config.intermediate_size)
            if new_intermediate_size != language_model.config.intermediate_size:
                print(f"Padding intermediate size: {language_model.config.intermediate_size} -> {new_intermediate_size}")
                for name, module in language_model.named_modules():
                    if (
                        "mlp.down_proj" in name
                        and "model.layers" in name
                        and isinstance(module, torch.nn.Linear)
                    ):
                        new_module = torch.nn.Linear(
                            new_intermediate_size,
                            module.out_features,
                            dtype=module.weight.dtype,
                        ).to(module.weight.device)
                        with torch.no_grad():
                            new_module.weight[:, : module.in_features] = module.weight.data
                            if module.bias is not None:
                                new_module.bias[: module.out_features].copy_(module.bias)
                        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                        if parent_name:
                            parent = dict(language_model.named_modules())[parent_name]
                            setattr(parent, name.split(".")[-1], new_module)
                        else:
                            setattr(language_model, name, new_module)
                language_model.config.intermediate_size = new_intermediate_size
                language_model.config.need_pad = True

        # Get rotation matrix
        Q = get_orthogonal_matrix(language_model.config.hidden_size, args.rotate_mode)

        config = language_model.config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads

        # Rotate embeddings AND aligner output (ensures both text and vision are rotated)
        print("  - Rotating embeddings and aligner output...")
        rotate_janus_embeddings(model, Q)

        # Rotate lm_head
        print("  - Rotating lm_head...")
        rotate_head(model, Q)
        utils.cleanup_memory()

        # Rotate each transformer layer
        for idx, layer in enumerate(
            tqdm.tqdm(language_model.model.layers, unit="layer", desc="Rotating Janus LLM")
        ):
            layer_device = next(layer.parameters()).device
            Q_layer = Q.to(layer_device)

            rotate_attention_inputs(layer, Q_layer)
            rotate_attention_output(layer, Q_layer)
            rotate_mlp_input(layer, Q_layer)
            rotate_mlp_output(layer, Q_layer, args.online_llm_hadamard)
            rotate_ov_proj(layer, num_heads, head_dim)

        utils.cleanup_memory()

    print("Janus model rotation complete.")
