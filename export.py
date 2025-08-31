#!/usr/bin/env python3
"""
Export a TorchScript Lite (PTL) depth model with Kornia preprocessing baked in.

Recommended package versions for PyTorch Mobile Lite 1.13.1:
  uv pip install --index-url https://download.pytorch.org/whl/cpu torch==1.13.1 torchvision==0.14.1
  uv pip install kornia==0.7.2 timm==0.6.7 numpy<2

Usage:
  uv venv && source .venv/bin/activate
  uv pip install --index-url https://download.pytorch.org/whl/cpu torch==1.13.1 torchvision==0.14.1
  uv pip install kornia==0.7.2 timm==0.6.7 numpy<2
  uv run python export.py --size 256 --out depth_kornia.torchscript.ptl
  # or select architecture and local weights
  uv run python export.py --model dpt_swin2_tiny_256 --weights weights/dpt_swin2_tiny_256.pt --size 256 --out depth_kornia.torchscript.ptl
"""

import argparse
import sys
from typing import Tuple, Optional

import torch
import torch.nn as nn

try:
    import numpy as np
    from packaging.version import Version
    if Version(np.__version__) >= Version("2.0.0"):
        print("ERROR: numpy>=2 detected. Install numpy<2 (e.g., uv pip install 'numpy<2').", file=sys.stderr)
        sys.exit(1)
except Exception:
    pass

try:
    import kornia as K
except Exception as e:
    print("Kornia is required. Install with: uv pip install kornia", file=sys.stderr)
    raise


MODEL_KEY_MAP = {
    "midas_small": "MiDaS_small",
    "dpt_swin2_tiny_256": "DPT_Swin2_Tiny_256",
    "dpt_levit_224": "DPT_LeViT_224",
    "midas_v21_small_256": "MiDaS_small",
}


class DepthKornia(nn.Module):
    """Wraps a depth net with Kornia preprocessing (resize + normalize)."""

    def __init__(self, net: nn.Module, in_h: int, in_w: int):
        super().__init__()
        self.net = net.eval()
        self.in_h = int(in_h)
        self.in_w = int(in_w)
        # ImageNet mean/std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1,3,H,W], RGB in [0,1] or [0,255]
        if x.dtype != torch.float32:
            x = x.float()
        if torch.isfinite(x).all():
            x = x / 255.0 if x.max() > 1.0 else x
        x = K.geometry.resize(
            x, (self.in_h, self.in_w), interpolation="bilinear", align_corners=False
        )
        x = (x - self.mean) / self.std
        y = self.net(x)  # [1,1,h,w] or [1,h,w]
        if y.dim() == 3:
            y = y.unsqueeze(1)
        return y


def load_midas_model(model_key: str, weights: Optional[str]) -> nn.Module:
    hub_key = MODEL_KEY_MAP.get(model_key.lower())
    if hub_key is None:
        print(f"Unknown --model {model_key}. Choices: {list(MODEL_KEY_MAP.keys())}", file=sys.stderr)
        sys.exit(2)
    net = torch.hub.load("intel-isl/MiDaS", hub_key, trust_repo=True, pretrained=False)
    if weights:
        sd = torch.load(weights, map_location="cpu")
        # Some weights wrap state in 'state_dict'
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        try:
            net.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"Warning: load_state_dict(strict=False) failed: {e}", file=sys.stderr)
    net.eval()
    return net


def export(model_name: str, weights_path: Optional[str], size: int, out_path: str) -> None:
    base = load_midas_model(model_name, weights_path)
    model = DepthKornia(base, size, size).eval()

    # Use tracing with strict=False and disabled check due to dynamic padding logic
    example = torch.randn(1, 3, size, size)
    with torch.no_grad():
        ts = torch.jit.trace(model, example, strict=False, check_trace=False)

    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        ts = optimize_for_mobile(ts)
    except Exception:
        pass

    ts._save_for_lite_interpreter(out_path)
    print(f"Wrote {out_path}")


def parse_args() -> Tuple[str, Optional[str], int, str]:
    p = argparse.ArgumentParser(description="Export TorchScript Lite depth model with Kornia preprocess")
    p.add_argument("--model", type=str, default="midas_small", help=f"Model key: {list(MODEL_KEY_MAP.keys())}")
    p.add_argument("--weights", type=str, default=None, help="Path to local .pt weights (optional)")
    p.add_argument("--size", type=int, default=256, help="Input resolution (HxW)")
    p.add_argument("--out", type=str, default="depth_kornia.torchscript.ptl", help="Output PTL path")
    a = p.parse_args()
    return a.model, a.weights, a.size, a.out


if __name__ == "__main__":
    m, w, s, o = parse_args()
    export(m, w, s, o)


