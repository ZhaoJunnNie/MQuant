"""TorchUMM-native calibration entry for Janus GPTQ in MQuant."""

from __future__ import annotations

import tqdm
import torch
import torch.nn as nn

from fake_quant import quant_utils
from fake_quant.gptq.gptq_utils import GPTQ
from fake_quant.gptq.janus_gptq_plus import janus_llm_rtn


@torch.no_grad()
def gptq_janus_fwrd_llm_torchumm(model, dataset, dev, args, quantizers):
    language_model = model.model.language_model
    layers = language_model.model.layers

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

    layers[0] = Catcher(layers[0])

    for sample in tqdm.tqdm(dataset.iter_samples(), desc="Collecting TorchUMM calibration data"):
        if cache["i"] >= args.nsamples:
            break
        try:
            model.generate(message=sample, dataset=None)
        except ValueError:
            pass

    layers[0] = layers[0].module
    print(f"Collected {cache['i']} calibration samples")

    for i in tqdm.tqdm(range(len(layers)), desc="(GPTQ Quant.) Janus LLM Layers"):
        layer = layers[i]
        subset = quant_utils.find_qlayers(layer, layers=[torch.nn.Linear])

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

        def add_batch(name):
            def tmp(_, inp, out):
                gptq_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gptq_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

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

        for handle in handles:
            handle.remove()

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
def janus_rtn_gptq_fwrd_torchumm(model, dataset, dev, args):
    quantizers = {}

    if args.quant_llm and args.llm_w_bits < 16:
        if args.llm_w_rtn:
            print("Using RTN for Janus LLM weight quantization")
            janus_llm_rtn(model, dev, args, quantizers)
        else:
            print("Using GPTQ for Janus LLM weight quantization")
            gptq_janus_fwrd_llm_torchumm(model, dataset, dev, args, quantizers)

    return quantizers
