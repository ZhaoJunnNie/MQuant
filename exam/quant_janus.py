"""
Janus Model Quantization with MQuant Framework

This script provides quantization support for Janus multimodal models
using the MQuant framework with VLMEvalKit for evaluation.

Environment Requirements:
========================
This script should be run in the appropriate conda environment with:
- janus package installed
- vlmeval package available

Usage:
  conda activate <env>
  python MQuant/exam/quant_janus.py --model_name Janus-Pro-1B [options]

Architecture Notes:
===================
Janus is a multimodal model with:
- Vision encoder (CLIP-based for understanding images)
- Aligner (projects vision features to language model space)
- Language model (LlamaForCausalLM - standard Llama transformer)

The quantization focuses on the language model component, which is a
standard LlamaForCausalLM with:
- Embedding layer (language_model.model.embed_tokens)
- Transformer decoder layers (language_model.model.layers[i])
  - Self-attention with q_proj, k_proj, v_proj, o_proj
  - MLP with gate_proj, up_proj, down_proj
  - Layer norms (input_layernorm, post_attention_layernorm)
- Output norm (language_model.model.norm)
- LM head (language_model.lm_head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import datetime
import os
import functools

from datasets import load_dataset
from loguru import logger
from evaluation.eval import eval_dataset
from fake_quant import quant_utils
from fake_quant import gptq
from fake_quant import utils
from fake_quant import hadamard_utils
from fake_quant.janus_rotation import fuse_janus_layer_norms, rotate_janus_model
from vlmeval.config import supported_VLM

torch.set_grad_enabled(False)


def init_logger(args):
    """Initialize logger with model name and timestamp."""
    logger_file = str(datetime.datetime.now().strftime("%m-%d %H:%M:%S")) + ".log"
    os.makedirs("log", exist_ok=True)
    if args.model_name is not None:
        logger_file = args.model_name + "_" + logger_file
    logger_file = "log/" + logger_file
    logger.add(logger_file)


# Model path settings
Model_Setting = {
    "Janus-1.3B": "deepseek-ai/Janus-1.3B",
    "Janus-Pro-1B": "deepseek-ai/Janus-Pro-1B",
    "Janus-Pro-7B": "deepseek-ai/Janus-Pro-7B",
}


def janus_add_act_quant(model, args):
    """
    Add activation quantization wrappers to Janus language model.

    Args:
        model: VLMEvalKit model wrapper
        args: Arguments with quant_llm flag
    """
    if args.quant_llm:
        print("Adding activation quantization to Janus LLM...")
        # Add ActQuantWrapper to all linear layers in language model
        for layer in model.model.language_model.model.layers:
            quant_utils.add_actquant(layer)
        print(f"Added activation quantization to {len(model.model.language_model.model.layers)} layers")


def calib_janus(model, args, dataset, calib_num):
    """
    Calibrate activation quantizers using dataset samples.

    Args:
        model: VLMEvalKit model wrapper
        args: Quantization arguments
        dataset: Calibration dataset
        calib_num: Number of calibration samples
    """
    import math
    from tqdm import tqdm

    lt = len(dataset.data)
    step = math.ceil(lt / calib_num)
    print(f"Calibrating Janus with {calib_num} samples (step={step})...")

    # Find all ActQuantWrapper layers
    qlayers = quant_utils.find_qlayers(
        model.model.language_model.model, layers=[quant_utils.ActQuantWrapper]
    )

    # Enable calibration mode
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.calibrate = True

    # Save and reduce max_new_tokens for faster calibration
    max_new_tokens = model.kwargs.get("max_new_tokens", 512)
    model.kwargs["max_new_tokens"] = 20

    # Run calibration forward passes
    for i in tqdm(range(0, lt, step), desc="Calibrating"):
        # On last sample, set last_calibrate to compute quantization params
        if i + step >= lt:
            print("Last calibration step - computing quantization params")
            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                qlayers[name].quantizer.last_calibrate = True
            model.kwargs["max_new_tokens"] = 1

        if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(args.dataset_name):
            struct = model.build_prompt(dataset.data.iloc[i], dataset=args.dataset_name)
        else:
            struct = dataset.build_prompt(dataset.data.iloc[i])

        try:
            model.generate(message=struct, dataset=args.dataset_name)
        except Exception as e:
            print(f"Warning: Calibration sample {i} failed: {e}")
            continue

    # Restore max_new_tokens
    model.kwargs["max_new_tokens"] = max_new_tokens

    # Disable calibration mode and enable quantization
    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.calibrate = False
        qlayers[name].quantizer.quant = True

    print("Calibration complete.")


def main(args):
    """Main quantization and evaluation function."""
    model_name = args.model_name

    # Check if model is supported
    if model_name not in Model_Setting:
        print(f"Error: Model {model_name} not found in Model_Setting.")
        print(f"Available models: {list(Model_Setting.keys())}")
        return

    # Load model via VLMEvalKit
    print(f"Loading {model_name}...")
    model = supported_VLM[model_name](
        model_path=Model_Setting[model_name], verbose=args.verbose
    )

    # Set random seed
    utils.seed_everything(args.seed)

    # Fuse layer norms if requested
    if not args.not_fuse_layer_norms:
        fuse_janus_layer_norms(model, args)

    # Apply rotation if requested
    if args.rotate:
        rotate_janus_model(model, args)

    # Handle online Hadamard rotation (rotation only, no quantization)
    if not args.quant and args.online_llm_hadamard:
        if args.rotate_llm:
            args.quant_llm = True
        janus_add_act_quant(model, args)
        qlayers = quant_utils.find_qlayers(
            model.model.language_model.model, layers=[quant_utils.ActQuantWrapper]
        )
        for name in qlayers:
            if "mlp.down_proj" in name:
                intermediate_size = model.model.language_model.config.intermediate_size
                had_K, K = hadamard_utils.get_hadK(intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
                if hasattr(model.model.language_model.config, 'need_pad') and model.model.language_model.config.need_pad:
                    hook = functools.partial(
                        utils.revise_down_input,
                        new_size=intermediate_size,
                    )
                    qlayers[name].register_forward_pre_hook(hook)

    # Full quantization mode
    if args.quant:
        # Configure quantization flags
        if args.online_llm_hadamard:
            if args.rotate_llm:
                args.quant_llm = True

        # Add activation quantization wrappers
        janus_add_act_quant(model, args)

        # Configure online Hadamard for down_proj layers
        if args.online_llm_hadamard and args.rotate_llm:
            print("Adding online Hadamard rotation to LLM...")
            qlayers = quant_utils.find_qlayers(
                model.model.language_model.model, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if "mlp.down_proj" in name:
                    intermediate_size = model.model.language_model.config.intermediate_size
                    had_K, K = hadamard_utils.get_hadK(intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                    qlayers[name].split = args.llm_split
                    if args.llm_split:
                        qlayers[name].split_weights()
                    if hasattr(model.model.language_model.config, 'need_pad') and model.model.language_model.config.need_pad:
                        hook = functools.partial(
                            utils.revise_down_input,
                            new_size=intermediate_size,
                        )
                        qlayers[name].register_forward_pre_hook(hook)

        # Weight quantization
        if args.load_gptq:
            print(f"Loading GPTQ model from: {args.load_gptq}")
            model.model = torch.load(args.load_gptq)
        else:
            from vlmeval.dataset import build_dataset

            dataset = build_dataset(args.dataset_name)
            model.set_dump_image(dataset.dump_image)

            # Apply RTN or GPTQ weight quantization
            quantizers = gptq.janus_rtn_gptq_fwrd_plus(
                model, dataset, utils.DEV, args.dataset_name, args
            )

            if args.dump_gptq:
                torch.save(model.model, args.dump_gptq)
                print(f"Dumped the GPTQ model to: {args.dump_gptq}")

        # Configure activation quantization for LLM
        if args.llm_a_bits < 16 or args.llm_static:
            if args.llm_static and args.llm_a_bits >= 16:
                print("Warning: If you want to run act with fp16, please set --static False")

            qlayers = quant_utils.find_qlayers(
                model.model.language_model.model, layers=[quant_utils.ActQuantWrapper]
            )
            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue

                layer_input_bits = args.llm_a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not (args.a_asym)
                layer_a_clip = args.a_clip_ratio

                qlayers[name].quantizer.configure(
                    bits=layer_input_bits,
                    groupsize=layer_groupsize,
                    sym=layer_a_sym,
                    clip_ratio=layer_a_clip,
                    act_per_tensor=args.act_per_tensor,
                    static=args.llm_static,
                    observer_type="minmax",
                )

    # Build dataset for evaluation
    from vlmeval.dataset import build_dataset
    dataset = build_dataset(args.dataset_name)
    model.set_dump_image(dataset.dump_image)

    # Calibrate static quantization if needed
    if args.llm_static:
        calib_janus(model, args, dataset, args.calib_num)

    # Set max_new_tokens if specified
    if args.max_new_tokens is not None:
        model.kwargs["max_new_tokens"] = args.max_new_tokens

    # Run evaluation
    eval_dataset(
        model,
        dataset,
        args.dataset_name,
        model_name=model_name,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Janus Model Quantization with MQuant")

    # Model arguments
    parser.add_argument("--model_name", type=str, default="Janus-Pro-1B",
                        choices=["Janus-1.3B", "Janus-Pro-1B", "Janus-Pro-7B"],
                        help="Janus model variant to use")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--quant", action="store_true", help="Enable quantization")

    # Rotation Arguments
    parser.add_argument("--rotate", action="store_true", default=False,
                        help="Rotate the model")
    parser.add_argument("--rotate_llm", action="store_true", default=False,
                        help="Rotate the LLM backbone")
    parser.add_argument("--rotate_mode", type=str, default="hadamard",
                        choices=["hadamard", "random"],
                        help="Rotation matrix type")

    # Activation Quantization Arguments
    parser.add_argument("--llm_a_bits", type=int, default=8,
                        help="Number of bits for LLM activation quantization")
    parser.add_argument("--a_groupsize", type=int, default=-1,
                        help="Groupsize for activation quantization")
    parser.add_argument("--a_asym", action="store_true", default=False,
                        help="Asymmetric activation quantization")
    parser.add_argument("--a_clip_ratio", type=float, default=1.0,
                        help="Clip ratio for activation quantization")

    # Weight Quantization Arguments
    parser.add_argument("--llm_w_bits", type=int, default=4,
                        help="Number of bits for LLM weight quantization")
    parser.add_argument("--w_groupsize", type=int, default=-1,
                        help="Groupsize for weight quantization")
    parser.add_argument("--w_asym", action="store_true", default=False,
                        help="Asymmetric weight quantization")
    parser.add_argument("--llm_w_rtn", action="store_true", default=False,
                        help="Use RTN for weight quantization instead of GPTQ")
    parser.add_argument("--llm_w_clip", action="store_true", default=False,
                        help="Enable weight clipping during quantization")
    parser.add_argument("--percdamp", type=float, default=0.01,
                        help="Percent dampening for GPTQ")
    parser.add_argument("--act_order", action="store_true", default=False,
                        help="Use activation ordering in GPTQ")

    # General Quantization Arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--int8_down_proj", action="store_true", default=False,
                        help="Use INT8 for down projection")
    parser.add_argument("--quant_llm", action="store_true", default=False,
                        help="Quantize the LLM backbone")
    parser.add_argument("--act_per_tensor", action="store_true", default=False,
                        help="Per-tensor activation quantization")
    parser.add_argument("--nsamples", type=int, default=8,
                        help="Number of calibration samples for GPTQ")
    parser.add_argument("--skip_names", nargs="+", default=[],
                        help="Skip quantization of layers with these names")

    # Fusion Arguments
    parser.add_argument("--no_fuse_llm", action="store_true", default=False,
                        help="Skip fusing LLM layer norms")
    parser.add_argument("--not_fuse_layer_norms", action="store_true", default=False,
                        help="Skip all layer norm fusion")

    # Static Quantization Arguments
    parser.add_argument("--llm_static", action="store_true", default=False,
                        help="Use static activation quantization for LLM")
    parser.add_argument("--calib_num", type=int, default=32,
                        help="Number of calibration samples")

    # Evaluation Arguments
    parser.add_argument("--eval_num", type=int, default=32,
                        help="Number of evaluation samples")
    parser.add_argument("--dataset_name", type=str, default="TextVQA_VAL",
                        help="Dataset name for evaluation")

    # Online Hadamard Arguments
    parser.add_argument("--online_llm_hadamard", action="store_true", default=False,
                        help="Enable online Hadamard rotation for LLM")
    parser.add_argument("--fp32_had", action="store_true", default=False,
                        help="Apply Hadamard rotation in FP32")
    parser.add_argument("--llm_split", action="store_true", default=False,
                        help="Split Hadamard rotation for LLM")

    # Save/Load Arguments
    parser.add_argument("--dump_gptq", type=str, default=None,
                        help="Path to save quantized model")
    parser.add_argument("--load_gptq", type=str, default=None,
                        help="Path to load quantized model")

    # Generation Arguments
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum new tokens for generation")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Verbose output during evaluation")

    args = parser.parse_args()
    init_logger(args)
    main(args)
