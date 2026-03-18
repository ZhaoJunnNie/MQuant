"""
Janus GPTQ Quantization Module for MQuant Framework

This module provides RTN and GPTQ weight quantization for Janus model's
language model backbone (LlamaForCausalLM).

Janus Architecture:
- model.language_model: LlamaForCausalLM
  - language_model.model.layers[i]: Transformer decoder layers
    - self_attn: q_proj, k_proj, v_proj, o_proj
    - mlp: gate_proj, up_proj, down_proj
"""

import math
import time
import tqdm
import torch
import torch.nn as nn
from fake_quant import utils
from fake_quant import quant_utils
from fake_quant.gptq.gptq_utils import GPTQ
import logging

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def janus_llm_rtn(model, dev, args, quantizers):
    """
    RTN (Round-to-Nearest) weight quantization for Janus language model.

    Args:
        model: VLMEvalKit model wrapper with model.model as Janus model
        dev: Target device
        args: Quantization arguments
        quantizers: Dictionary to store quantizers
    """
    language_model = model.model.language_model
    layers = language_model.model.layers

    for i in tqdm.tqdm(
        range(len(layers)), desc="(RtN Quant.) Janus LLM Layers"
    ):
        layer = layers[i]

        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            if any(p_name in name for p_name in args.skip_names):
                continue

            layer_weight_bits = args.llm_w_bits
            quantizer = quant_utils.WeightQuantizer()
            quantizer.configure(
                layer_weight_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.llm_w_clip,
            )
            W = subset[name].weight.data
            dtype = W.dtype
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(dtype)
            quantizers["model.language_model.model.layers.%d.%s" % (i, name)] = quantizer.cpu()

        torch.cuda.empty_cache()


@torch.no_grad()
def gptq_janus_fwrd_llm(
    model, dataset, dev, dataset_name, args, quantizers
):
    """
    GPTQ weight quantization for Janus language model.

    Uses calibration data to compute Hessian for optimal quantization.

    Args:
        model: VLMEvalKit model wrapper
        dataset: Calibration dataset
        dev: Target device
        dataset_name: Name of the dataset
        args: Quantization arguments
        quantizers: Dictionary to store quantizers
    """
    language_model = model.model.language_model
    layers = language_model.model.layers

    # Cache for storing intermediate activations
    inps = [None] * args.nsamples
    attention_masks = [None] * args.nsamples
    position_ids = [None] * args.nsamples
    position_embeddings = [None] * args.nsamples
    cache_positions = [None] * args.nsamples
    cache = {"i": 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            inps[cache["i"]] = hidden_states
            attention_masks[cache["i"]] = kwargs.get("attention_mask", None)
            position_ids[cache["i"]] = kwargs.get("position_ids", None)
            position_embeddings[cache["i"]] = kwargs.get("position_embeddings", None)
            cache_positions[cache["i"]] = kwargs.get("cache_position", None)
            cache["i"] += 1
            raise ValueError

    # Wrap first layer to capture inputs
    layers[0] = Catcher(layers[0])

    # Run forward passes to collect calibration data
    lt = len(dataset.data)
    for i in tqdm.tqdm(range(lt), desc="Collecting calibration data"):
        if cache["i"] >= args.nsamples:
            break

        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=dataset_name)
        except ValueError:
            pass

    # Restore original layer
    layers[0] = layers[0].module

    print(f"Collected {cache['i']} calibration samples")

    # Process each layer with GPTQ
    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQ Quant.) Janus LLM Layers"):
        layer = layers[i]
        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

        # Create GPTQ instances for each linear layer
        gptq_layers = {}
        for name in subset:
            if any(p_name in name for p_name in args.skip_names):
                continue

            gptq_layers[name] = GPTQ(subset[name])
            gptq_layers[name].quantizer = quant_utils.WeightQuantizer()
            gptq_layers[name].quantizer.configure(
                args.llm_w_bits,
                perchannel=True,
                sym=not (args.w_asym),
                mse=args.llm_w_clip,
            )

        # Register hooks to collect batch statistics
        def add_batch(name):
            def tmp(_, inp, out):
                gptq_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # Forward pass through this layer for all samples
        for j in range(args.nsamples):
            if inps[j] is None:
                continue
            kwargs = {}
            if attention_masks[j] is not None:
                kwargs["attention_mask"] = attention_masks[j]
            if position_ids[j] is not None:
                kwargs["position_ids"] = position_ids[j]
            if position_embeddings[j] is not None:
                kwargs["position_embeddings"] = position_embeddings[j]
            if cache_positions[j] is not None:
                kwargs["cache_position"] = cache_positions[j]
            layer(inps[j], **kwargs)

        # Remove hooks
        for h in handles:
            h.remove()

        # Quantize weights
        for name in gptq_layers:
            gptq_layers[name].fasterquant(
                percdamp=args.percdamp,
                groupsize=args.w_groupsize,
                actorder=args.act_order,
                static_groups=False,
            )
            quantizers["model.language_model.model.layers.%d.%s" % (i, name)] = (
                gptq_layers[name].quantizer.cpu()
            )
            gptq_layers[name].free()

        # Update inputs for next layer
        for j in range(args.nsamples):
            if inps[j] is None:
                continue
            kwargs = {}
            if attention_masks[j] is not None:
                kwargs["attention_mask"] = attention_masks[j]
            if position_ids[j] is not None:
                kwargs["position_ids"] = position_ids[j]
            if position_embeddings[j] is not None:
                kwargs["position_embeddings"] = position_embeddings[j]
            if cache_positions[j] is not None:
                kwargs["cache_position"] = cache_positions[j]
            inps[j] = layer(inps[j], **kwargs)[0]

        torch.cuda.empty_cache()

    print("-----GPTQ Quantization Janus LLM Done-----")


@torch.no_grad()
def janus_rtn_gptq_fwrd_plus(
    model, dataset, dev, dataset_name, args
):
    """
    Main entry point for Janus weight quantization.

    Supports both RTN and GPTQ methods based on args.

    Args:
        model: VLMEvalKit model wrapper
        dataset: Calibration dataset
        dev: Target device
        dataset_name: Name of the dataset
        args: Quantization arguments

    Returns:
        Dictionary of quantizers
    """
    quantizers = {}

    # Quantize language model weights
    if args.quant_llm:
        if args.llm_w_bits < 16:
            if args.llm_w_rtn:
                print("Using RTN for Janus LLM weight quantization")
                janus_llm_rtn(model, dev, args, quantizers)
            else:
                print("Using GPTQ for Janus LLM weight quantization")
                gptq_janus_fwrd_llm(
                    model, dataset, dev, dataset_name, args, quantizers
                )

    return quantizers
