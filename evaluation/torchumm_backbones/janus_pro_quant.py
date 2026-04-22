"""MQuant Janus-Pro quantized backbone for TorchUMM."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_TORCHUMM_SRC = Path(__file__).resolve().parents[3] / "TorchUMM" / "src"
if str(_TORCHUMM_SRC) not in sys.path:
    sys.path.insert(0, str(_TORCHUMM_SRC))

from umm.backbones.janus_pro.adapter import JanusProBackbone


class MQuantJanusProBackbone(JanusProBackbone):
    """Pre-quantized MQuant Janus-Pro backbone."""

    name = "janus_pro_quant_mquant"
    _build_model_call_count: int = 0

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._canonical_cfg: dict[str, Any] | None = None
        self._loaded = False

    def inject_model(
        self,
        vl_chat_processor: Any,
        tokenizer: Any,
        model: Any,
        canonical_cfg: dict[str, Any],
    ) -> None:
        """Set the already-loaded, already-quantized model."""
        self.vl_chat_processor = vl_chat_processor
        self.tokenizer = tokenizer
        self.model = model
        self._canonical_cfg = canonical_cfg
        self._loaded = True

        if "model_path" in canonical_cfg:
            self.model_path = canonical_cfg["model_path"]
        if "janus_root" in canonical_cfg:
            self.janus_root = Path(canonical_cfg["janus_root"]).expanduser()
        if "seed" in canonical_cfg:
            self.seed = int(canonical_cfg["seed"])
        if "torch_dtype" in canonical_cfg:
            self.torch_dtype = str(canonical_cfg["torch_dtype"])

        generation_cfg = canonical_cfg.get("generation_cfg")
        if isinstance(generation_cfg, dict):
            self.default_generation_cfg.update(generation_cfg)
        understanding_cfg = canonical_cfg.get("understanding_cfg")
        if isinstance(understanding_cfg, dict):
            self.default_understanding_cfg.update(understanding_cfg)

    def load(self, cfg: dict[str, Any]) -> None:
        """Guarded no-op after injection; validates config consistency."""
        if (
            not self._loaded
            or self.vl_chat_processor is None
            or self.tokenizer is None
            or self.model is None
        ):
            raise RuntimeError(
                "MQuantJanusProBackbone.load() called before inject_model() "
                "provided model, processor, and tokenizer state."
            )

        check_keys = ("model_path", "janus_root", "torch_dtype", "seed")
        for key in check_keys:
            expected = (self._canonical_cfg or {}).get(key)
            actual = cfg.get(key)
            if actual is not None and expected is not None and str(actual) != str(expected):
                raise ValueError(
                    f"MQuantJanusProBackbone.load() cfg mismatch on {key!r}: "
                    f"canonical={expected!r}, incoming={actual!r}. "
                    "A second load with different config was attempted."
                )

    def _build_model(self):
        MQuantJanusProBackbone._build_model_call_count += 1
        if MQuantJanusProBackbone._build_model_call_count > 1:
            raise RuntimeError(
                f"_build_model called {MQuantJanusProBackbone._build_model_call_count} "
                "times; the singleton guarantee is broken."
            )
        return super()._build_model()


def make_mquant_janus_pro_backbone(
    model: Any,
    vl_chat_processor: Any,
    tokenizer: Any,
    canonical_cfg: dict[str, Any],
) -> MQuantJanusProBackbone:
    """Create a singleton MQuant Janus-Pro backbone around an injected model."""
    bb = MQuantJanusProBackbone()
    bb.inject_model(vl_chat_processor, tokenizer, model, canonical_cfg)
    return bb
