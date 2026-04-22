#!/usr/bin/env python3
"""MQuant Janus-Pro quantization with TorchUMM MMMU evaluation."""

from __future__ import annotations

import argparse
import copy
import datetime
import functools
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MQUANT_ROOT = Path(__file__).resolve().parents[1]
_TORCHUMM_SRC = _REPO_ROOT / "TorchUMM" / "src"


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path.exists() and path_str not in sys.path:
        sys.path.insert(0, path_str)


for _path in (_TORCHUMM_SRC, _MQUANT_ROOT):
    _prepend_sys_path(_path)

from fake_quant import hadamard_utils, quant_utils, utils
from fake_quant.gptq.janus_gptq_torchumm import janus_rtn_gptq_fwrd_torchumm
from fake_quant.janus_rotation import fuse_janus_layer_norms, rotate_janus_model
from janus_torchumm_runtime import (
    build_torchumm_calibration_dataset,
    load_torchumm_janus_runtime,
)

torch.set_grad_enabled(False)


def init_logger(args: argparse.Namespace) -> None:
    logger_file = str(datetime.datetime.now().strftime("%m-%d %H:%M:%S")) + ".log"
    os.makedirs("log", exist_ok=True)
    if args.model_name is not None:
        logger_file = args.model_name + "_" + logger_file
    logger_file = "log/" + logger_file
    logger.add(logger_file)


Model_Setting = {
    "Janus-1.3B": "deepseek-ai/Janus-1.3B",
    "Janus-Pro-1B": "deepseek-ai/Janus-Pro-1B",
    "Janus-Pro-7B": "deepseek-ai/Janus-Pro-7B",
}


def janus_add_act_quant(model: Any, args: argparse.Namespace) -> None:
    if args.quant_llm:
        print("Adding activation quantization to Janus LLM...")
        for layer in model.model.language_model.model.layers:
            quant_utils.add_actquant(layer)
        print(f"Added activation quantization to {len(model.model.language_model.model.layers)} layers")


def calib_janus(model: Any, args: argparse.Namespace, dataset: Any, calib_num: int) -> None:
    from tqdm import tqdm

    samples = list(dataset.iter_samples())
    if not samples:
        raise RuntimeError("TorchUMM calibration dataset is empty.")
    selected = samples[:max(1, min(calib_num, len(samples)))]
    print(f"Calibrating Janus with {len(selected)} TorchUMM samples...")

    qlayers = quant_utils.find_qlayers(
        model.model.language_model.model, layers=[quant_utils.ActQuantWrapper]
    )

    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.calibrate = True

    max_new_tokens = model.kwargs.get("max_new_tokens", 512)
    model.kwargs["max_new_tokens"] = 20

    for i, sample in enumerate(tqdm(selected, desc="Calibrating")):
        if i == len(selected) - 1:
            print("Last calibration step - computing quantization params")
            for name in qlayers:
                if any(p_name in name for p_name in args.skip_names):
                    continue
                qlayers[name].quantizer.last_calibrate = True
            model.kwargs["max_new_tokens"] = 1

        try:
            model.generate(message=sample, dataset=None)
        except Exception as exc:
            print(f"Warning: Calibration sample {i} failed: {exc}")
            continue

    model.kwargs["max_new_tokens"] = max_new_tokens

    for name in qlayers:
        if any(p_name in name for p_name in args.skip_names):
            continue
        qlayers[name].quantizer.calibrate = False
        qlayers[name].quantizer.quant = True

    print("Calibration complete.")


def _load_canonical_cfg(args: argparse.Namespace, model_name: str) -> dict[str, Any]:
    from umm.core.config import load_config

    raw_cfg = load_config(args.mmmu_config)
    inference_cfg = raw_cfg.get("inference", {})
    if not isinstance(inference_cfg, dict):
        raise ValueError("`inference` must be a mapping in the MMMU config.")

    backbone_name = inference_cfg.get("backbone")
    if str(backbone_name) != "janus_pro_quant_mquant":
        raise ValueError(
            "`inference.backbone` must be `janus_pro_quant_mquant`, "
            f"got {backbone_name!r}."
        )

    backbone_cfg = inference_cfg.get("backbone_cfg", {})
    if not isinstance(backbone_cfg, dict):
        raise ValueError("`inference.backbone_cfg` must be a mapping in the MMMU config.")

    required_keys = ("model_path", "seed", "torch_dtype", "understanding_cfg")
    missing = [key for key in required_keys if key not in backbone_cfg]
    if missing:
        raise ValueError(f"`inference.backbone_cfg` missing required keys: {missing}")

    expected_model_path = Model_Setting[model_name]
    if str(backbone_cfg["model_path"]) != str(expected_model_path):
        raise ValueError(
            "`inference.backbone_cfg.model_path` must match the selected "
            f"model_name {model_name!r}: expected {expected_model_path!r}, "
            f"got {backbone_cfg['model_path']!r}."
        )
    if str(backbone_cfg["seed"]) != str(args.seed):
        raise ValueError(
            "`inference.backbone_cfg.seed` must match `--seed`: "
            f"expected {args.seed!r}, got {backbone_cfg['seed']!r}."
        )
    if str(backbone_cfg["torch_dtype"]).lower() not in {"bf16", "bfloat16"}:
        raise ValueError(
            "MQuant's Janus wrapper loads bfloat16; "
            f"got torch_dtype={backbone_cfg['torch_dtype']!r}."
        )
    if not isinstance(backbone_cfg["understanding_cfg"], dict):
        raise ValueError("`inference.backbone_cfg.understanding_cfg` must be a mapping.")

    canonical_cfg = copy.deepcopy(backbone_cfg)
    canonical_cfg["model_path"] = expected_model_path
    canonical_cfg["seed"] = args.seed
    canonical_cfg["torch_dtype"] = str(backbone_cfg["torch_dtype"])
    if canonical_cfg != backbone_cfg:
        raise ValueError(
            "The driver canonical cfg must match `inference.backbone_cfg` exactly. "
            f"canonical={canonical_cfg!r}, yaml={backbone_cfg!r}."
        )
    return canonical_cfg


def _normalize_aliases(args: argparse.Namespace) -> None:
    args.w_bits = args.llm_w_bits
    args.a_bits = args.llm_a_bits
    args.w_rtn = args.llm_w_rtn
    args.w_clip = args.llm_w_clip


def main(args: argparse.Namespace) -> int:
    model_name = args.model_name

    if model_name not in Model_Setting:
        print(f"Error: Model {model_name} not found in Model_Setting.")
        print(f"Available models: {list(Model_Setting.keys())}")
        return 1

    canonical_cfg = _load_canonical_cfg(args, model_name)

    print(f"Loading {model_name} through TorchUMM...")
    model = load_torchumm_janus_runtime(canonical_cfg)

    utils.seed_everything(args.seed)

    if not args.not_fuse_layer_norms:
        fuse_janus_layer_norms(model, args)

    if args.rotate:
        rotate_janus_model(model, args)

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
                if hasattr(model.model.language_model.config, "need_pad") and model.model.language_model.config.need_pad:
                    hook = functools.partial(
                        utils.revise_down_input,
                        new_size=intermediate_size,
                    )
                    qlayers[name].register_forward_pre_hook(hook)

    if args.quant:
        if args.online_llm_hadamard:
            if args.rotate_llm:
                args.quant_llm = True

        janus_add_act_quant(model, args)

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
                    if hasattr(model.model.language_model.config, "need_pad") and model.model.language_model.config.need_pad:
                        hook = functools.partial(
                            utils.revise_down_input,
                            new_size=intermediate_size,
                        )
                        qlayers[name].register_forward_pre_hook(hook)

        if args.load_gptq:
            print(f"Loading GPTQ model from: {args.load_gptq}")
            model.set_model(torch.load(args.load_gptq))
        else:
            dataset = build_torchumm_calibration_dataset(
                args.mmmu_config,
                max_samples=max(args.nsamples, args.calib_num),
            )

            quantizers = janus_rtn_gptq_fwrd_torchumm(model, dataset, utils.DEV, args)

            if args.dump_gptq:
                torch.save(model.model, args.dump_gptq)
                print(f"Dumped the GPTQ model to: {args.dump_gptq}")

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

    dataset = build_torchumm_calibration_dataset(
        args.mmmu_config,
        max_samples=max(args.nsamples, args.calib_num),
    )

    if args.llm_static:
        calib_janus(model, args, dataset, args.calib_num)

    if args.max_new_tokens is not None:
        model.kwargs["max_new_tokens"] = args.max_new_tokens

    vl_chat_processor = model.vl_chat_processor
    tokenizer = model.tokenizer
    raw_model = model.model
    if torch.cuda.is_available():
        raw_model = raw_model.cuda().eval()
        model.set_model(raw_model)

    from evaluation.torchumm_backbones import (
        MQuantJanusProBackbone,
        make_mquant_janus_pro_backbone,
    )
    from umm.core import registry
    from umm.cli.mmmu_eval import run_mmmu_eval_command

    bb = make_mquant_janus_pro_backbone(
        raw_model, vl_chat_processor, tokenizer, canonical_cfg
    )
    registry.register("backbone", "janus_pro_quant_mquant", lambda: bb)

    ret = 0
    try:
        ret = run_mmmu_eval_command(SimpleNamespace(config=args.mmmu_config))
    finally:
        print(
            "[guard] MQuantJanusProBackbone._build_model_call_count="
            f"{MQuantJanusProBackbone._build_model_call_count}"
        )

    assert MQuantJanusProBackbone._build_model_call_count <= 1, (
        "Singleton guarantee broken: stock Janus-Pro build ran more than once."
    )
    return int(ret)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Janus Model Quantization with MQuant and TorchUMM")

    parser.add_argument("--model_name", type=str, default="Janus-Pro-1B",
                        choices=["Janus-1.3B", "Janus-Pro-1B", "Janus-Pro-7B"],
                        help="Janus model variant to use")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--quant", action="store_true", help="Enable quantization")

    parser.add_argument("--rotate", action="store_true", default=False,
                        help="Rotate the model")
    parser.add_argument("--rotate_llm", action="store_true", default=False,
                        help="Rotate the LLM backbone")
    parser.add_argument("--rotate_mode", type=str, default="hadamard",
                        choices=["hadamard", "random"],
                        help="Rotation matrix type")

    parser.add_argument("--llm_a_bits", "--a_bits", dest="llm_a_bits", type=int, default=8,
                        help="Number of bits for LLM activation quantization")
    parser.add_argument("--a_groupsize", type=int, default=-1,
                        help="Groupsize for activation quantization")
    parser.add_argument("--a_asym", action="store_true", default=False,
                        help="Asymmetric activation quantization")
    parser.add_argument("--a_clip_ratio", type=float, default=1.0,
                        help="Clip ratio for activation quantization")

    parser.add_argument("--llm_w_bits", "--w_bits", dest="llm_w_bits", type=int, default=4,
                        help="Number of bits for LLM weight quantization")
    parser.add_argument("--w_groupsize", type=int, default=-1,
                        help="Groupsize for weight quantization")
    parser.add_argument("--w_asym", action="store_true", default=False,
                        help="Asymmetric weight quantization")
    parser.add_argument("--llm_w_rtn", "--w_rtn", dest="llm_w_rtn",
                        action="store_true", default=False,
                        help="Use RTN for weight quantization instead of GPTQ")
    parser.add_argument("--llm_w_clip", "--w_clip", dest="llm_w_clip",
                        action="store_true", default=False,
                        help="Enable weight clipping during quantization")
    parser.add_argument("--percdamp", type=float, default=0.01,
                        help="Percent dampening for GPTQ")
    parser.add_argument("--act_order", action="store_true", default=False,
                        help="Use activation ordering in GPTQ")

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

    parser.add_argument("--no_fuse_llm", action="store_true", default=False,
                        help="Skip fusing LLM layer norms")
    parser.add_argument("--not_fuse_layer_norms", action="store_true", default=False,
                        help="Skip all layer norm fusion")

    parser.add_argument("--llm_static", action="store_true", default=False,
                        help="Use static activation quantization for LLM")
    parser.add_argument("--calib_num", type=int, default=32,
                        help="Number of calibration samples")

    parser.add_argument("--eval_num", type=int, default=32,
                        help="Number of evaluation samples")
    parser.add_argument("--dataset_name", type=str, default="TextVQA_VAL",
                        help="Deprecated; calibration follows the TorchUMM eval config")

    parser.add_argument("--online_llm_hadamard", action="store_true", default=False,
                        help="Enable online Hadamard rotation for LLM")
    parser.add_argument("--fp32_had", action="store_true", default=False,
                        help="Apply Hadamard rotation in FP32")
    parser.add_argument("--llm_split", action="store_true", default=False,
                        help="Split Hadamard rotation for LLM")

    parser.add_argument("--dump_gptq", type=str, default=None,
                        help="Path to save quantized model")
    parser.add_argument("--load_gptq", type=str, default=None,
                        help="Path to load quantized model")

    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum new tokens for generation")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Verbose output during evaluation")

    parser.add_argument("--mmmu_config", type=str, required=True,
                        help="Path to the MQuant TorchUMM MMMU YAML config")
    return parser


if __name__ == "__main__":
    parsed_args = build_parser().parse_args()
    _normalize_aliases(parsed_args)
    init_logger(parsed_args)
    raise SystemExit(main(parsed_args))
