import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Any

class PropertyHead(nn.Module):
    """
    Matches your training script:
    in_dim -> 256 -> 64 -> out_dim, ReLU + Dropout(p_drop)
    """
    def __init__(self, in_dim: int, out_dim: int, hidden1: int = 256, hidden2: int = 64, p_drop: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(hidden2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def load_property_head_checkpoint(
    ckpt_path: str,
    device: torch.device = torch.device("cpu"),
) -> Tuple[PropertyHead, Dict[str, torch.Tensor], List[str], Dict[str, Any]]:
    """
    Loads:
      - model_state
      - x_mu, x_sd, y_mu, y_sd  (scalers)
      - props (list[str])       (order of outputs)
      - latent_dim, arch        (metadata)
    Returns model (eval), scalers dict, props list, meta dict.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    props: List[str] = ckpt.get("props", [])
    if not props:
        raise ValueError("Checkpoint missing 'props' list; cannot map outputs to names.")

    latent_dim: int = int(ckpt.get("latent_dim", 0))
    if latent_dim <= 0:
        raise ValueError("Checkpoint missing/invalid 'latent_dim'.")

    arch = ckpt.get("arch", {"hidden1": 256, "hidden2": 64, "p_drop": 0.05})
    hidden1 = int(arch.get("hidden1", 256))
    hidden2 = int(arch.get("hidden2", 64))
    p_drop = float(arch.get("p_drop", 0.05))

    model = PropertyHead(in_dim=latent_dim, out_dim=len(props), hidden1=hidden1, hidden2=hidden2, p_drop=p_drop)
    state = ckpt.get("model_state", ckpt.get("state_dict", None))
    if state is None:
        raise ValueError("Checkpoint missing 'model_state' or 'state_dict'.")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    scalers = {
        "x_mu": ckpt["x_mu"].to(device).float(),  # shape [1, latent_dim]
        "x_sd": ckpt["x_sd"].to(device).float(),  # shape [1, latent_dim]
        "y_mu": ckpt["y_mu"].to(device).float(),  # shape [1, out_dim]
        "y_sd": ckpt["y_sd"].to(device).float(),  # shape [1, out_dim]
    }

    meta = {
        "best_val_mse_stdspace": float(ckpt.get("best_val_mse_stdspace", 0.0)),
        "val_metrics": ckpt.get("val_metrics", {}),
        "arch": arch,
    }
    return model, scalers, props, meta