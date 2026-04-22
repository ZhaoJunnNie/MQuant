"""TorchUMM-native Janus runtime and calibration data helpers for MQuant."""

from __future__ import annotations

from typing import Any


class TorchUMMJanusRuntime:
    """Small compatibility wrapper around a TorchUMM JanusProBackbone."""

    def __init__(self, backbone: Any) -> None:
        self.backbone = backbone
        self.model = backbone.model
        self.vl_chat_processor = backbone.vl_chat_processor
        self.tokenizer = backbone.tokenizer
        self.kwargs = dict(getattr(backbone, "default_understanding_cfg", {}))

    def set_model(self, model: Any) -> None:
        self.model = model
        self.backbone.model = model

    def generate(self, message: dict[str, Any], dataset: str | None = None) -> Any:
        if not isinstance(message, dict):
            raise TypeError("TorchUMM Janus calibration expects message as a sample dict.")
        prompt = message.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("TorchUMM Janus calibration sample requires a non-empty prompt.")
        images = message.get("images", [])
        if images is None:
            images = []
        return self.backbone.understanding(
            prompt=prompt,
            images=list(images),
            understanding_cfg=dict(self.kwargs),
        )


def load_torchumm_janus_runtime(backbone_cfg: dict[str, Any]) -> TorchUMMJanusRuntime:
    from umm.backbones.janus_pro import JanusProBackbone

    backbone = JanusProBackbone()
    backbone.load(dict(backbone_cfg))
    return TorchUMMJanusRuntime(backbone)


def build_torchumm_calibration_dataset(
    config_path: str,
    max_samples: int,
) -> Any:
    from umm.calibration import build_calibration_dataset

    return build_calibration_dataset(config_path, max_samples=max_samples)
