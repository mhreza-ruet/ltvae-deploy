import os, json, time, logging
from typing import Dict
from urllib.parse import quote

import torch

from .s3_utils import maybe_download_from_s3
from .smiles_utils import validate_smiles, smiles_png_base64
from .property_head import load_property_head_checkpoint

# ------------------------- logging -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ltvae.model")

# ------------------------- Vocab & tokenization -------------------------
def _load_vocab(vocab_path: str | None):
    """Load token_to_idx from path or common fallbacks."""
    if vocab_path and os.path.exists(vocab_path):
        with open(vocab_path, "r") as f:
            return json.load(f)
    for p in ("app/token_to_idx.json", "app/vocab.json", "token_to_idx.json", "vocab.json"):
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    raise FileNotFoundError("Vocabulary JSON not found. Set VOCAB_JSON or place app/token_to_idx.json")

def _tok(smiles: str, tok2idx: Dict[str, int], sos: int, eos: int, pad: int, max_len: int = 160):
    """
    Very simple tokenizer for demo: char-level mapping into ids with <SOS>/<EOS>/padding>.
    Replace with your real tokenizer if needed.
    """
    ids = [sos]
    for ch in smiles:
        ids.append(tok2idx.get(ch, tok2idx.get("<UNK>", eos)))
        if len(ids) >= max_len - 1:
            break
    ids.append(eos)
    ids = ids + [pad] * (max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)

# ------------------------- Manager -------------------------
class ModelManager:
    def __init__(self):
        # Paths
        self._model_path = os.getenv("MODEL_PATH", "app/checkpoints/best_model.pth")
        self._prop_path  = os.getenv("PROP_HEAD_PATH", "app/checkpoints/property_head_best.pt")
        self._vocab_path = os.getenv("VOCAB_JSON")  # fallbacks handled in _load_vocab()

        # ---- Match your checkpoint cfg exactly (env can override if set) ----
        self._d_model    = int(os.getenv("D_MODEL", "256"))
        self._latent_dim = int(os.getenv("LATENT_DIM", "64"))
        self._nhead      = int(os.getenv("NHEAD", "8"))
        self._enc_layers = int(os.getenv("ENC_LAYERS", "7"))
        self._dec_layers = int(os.getenv("DEC_LAYERS", "7"))
        self._ff_dim     = int(os.getenv("FF_DIM", "1024"))
        self._dropout    = float(os.getenv("DROPOUT", "0.05"))
        self._max_len    = int(os.getenv("SEQ_LENGTH", "160"))  # cfg["seq_length"]

        # Special token indices (assert against vocab)
        self._pad_idx = int(os.getenv("PAD_IDX", "0"))
        self._sos_idx = int(os.getenv("SOS_IDX", "2"))
        self._eos_idx = int(os.getenv("EOS_IDX", "3"))

        # State
        self._tok2idx: Dict[str, int] | None = None
        self._model = None            # LTVAE
        self._prop_head = None        # property head (optional)
        self._scalers = None          # scalers dict for prop head
        self._props = None            # property names order
        self._prop_meta = {}
        self._is_real = False
        self._has_prop = False
        self._last_error = None

    def is_loaded(self) -> bool:
        return self._model is not None

    # --------- Load LTVAE + property head ----------
    def _load_real_model(self) -> bool:
        try:
            logger.info("[ModelManager] Loading real model from %s", self._model_path)
            model_path = maybe_download_from_s3(self._model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"MODEL_PATH not found: {model_path}")

            # vocab
            logger.info("[ModelManager] Loading vocab from %s", self._vocab_path or "auto-fallback")
            self._tok2idx = _load_vocab(self._vocab_path)
            # assert special token indices match your cfg
            assert self._tok2idx["<PAD>"] == self._pad_idx, "PAD index mismatch with vocab"
            assert self._tok2idx["<SOS>"] == self._sos_idx, "SOS index mismatch with vocab"
            assert self._tok2idx["<EOS>"] == self._eos_idx, "EOS index mismatch with vocab"
            vocab_size = max(self._tok2idx.values()) + 1
            logger.info("[ModelManager] Vocab size: %d", vocab_size)

            # import your architecture
            from .ltvae_arch import LSTM_VAE_Trans

            device = torch.device("cpu")
            m = LSTM_VAE_Trans(
                vocab_size=vocab_size,
                d_model=self._d_model,
                latent_dim=self._latent_dim,
                pad_idx=self._pad_idx,
                sos_idx=self._sos_idx,
                eos_idx=self._eos_idx,
                enc_layers=self._enc_layers,
                dec_layers=self._dec_layers,
                nhead=self._nhead,
                dropout=self._dropout,
                max_len=self._max_len,
                dim_feedforward=self._ff_dim,  # must match cfg["ff_dim"]
            )

            sd = torch.load(model_path, map_location=device)
            state_dict = sd.get("state_dict", sd)
            m.load_state_dict(state_dict, strict=False)
            m.eval()
            logger.info("[ModelManager] LTVAE weights loaded (strict=False).")

            # property head (optional)
            has_prop = False
            prop_meta = {}
            if os.path.exists(self._prop_path):
                logger.info("[ModelManager] Loading property head from %s", self._prop_path)
                ph, scalers, props, meta = load_property_head_checkpoint(self._prop_path, device=device)
                self._prop_head = ph.eval()
                self._scalers = scalers
                self._props = props
                prop_meta = meta
                has_prop = True
                logger.info("[ModelManager] Property head loaded with props: %s", props)

            self._model = m
            self._is_real = True
            self._has_prop = has_prop
            self._prop_meta = prop_meta
            self._last_error = None
            return True

        except Exception as e:
            self._last_error = str(e)
            logger.exception("[ModelManager] Real model load failed: %s", e)
            return False

    # --------- Mock fallback ----------
    def _load_mock_model(self):
        logger.warning("[ModelManager] Using mock model (no real checkpoint).")
        class MockModel:
            def encode_to_latent(self, smiles: str, latent_dim: int):
                vals = [(hash(smiles + str(i)) % 200 - 100) / 100.0 for i in range(latent_dim)]
                return torch.tensor([vals], dtype=torch.float)

        self._model = MockModel()
        self._prop_head = None
        self._scalers = None
        self._props = None
        self._is_real = False
        self._has_prop = False
        self._prop_meta = {}

    def _ensure_loaded(self):
        if self._model is not None:
            return
        logger.info("[ModelManager] Ensuring model is loaded…")
        if self._load_real_model():
            logger.info("[ModelManager] Real model loaded.")
            return
        logger.warning("[ModelManager] Falling back to mock model.")
        self._load_mock_model()

    # --------- Latent inference ----------
    def _infer_latent(self, smiles: str) -> torch.Tensor:
        if self._is_real:
            x = _tok(smiles, self._tok2idx, self._sos_idx, self._eos_idx, self._pad_idx, max_len=self._max_len)
            memory, mem_pad, mu, logvar = self._model._encode(x)
            z = self._model.reparameterize(mu, logvar)  # [1, latent_dim]
            return z
        else:
            return self._model.encode_to_latent(smiles, self._latent_dim)

    # --------- Property inference (with scalers) ----------
    def _infer_properties(self, z: torch.Tensor) -> Dict[str, float]:
        if self._has_prop and self._prop_head is not None and self._scalers is not None:
            x_mu = self._scalers["x_mu"]
            x_sd = self._scalers["x_sd"]
            y_mu = self._scalers["y_mu"]
            y_sd = self._scalers["y_sd"]
            z_std = (z - x_mu) / (x_sd + 1e-8)
            with torch.no_grad():
                y_std = self._prop_head(z_std)
                y = y_std * y_sd + y_mu
                vals = y.squeeze(0).tolist()
            return {name: float(vals[i]) for i, name in enumerate(self._props)}

        # fallback (keeps API usable without prop head)
        vals = z.squeeze(0).tolist()
        base = sum(vals[:5]) if len(vals) >= 5 else sum(vals)
        return {"AromRings": 0.0, "MW": round(100.0 + 10.0 * base, 2), "LogP": 0.0, "Fsp3": 0.5, "QED": 0.5}

    # --------- Public predict ----------
    def predict(self, smiles: str):
        if not isinstance(smiles, str) or len(smiles.strip()) == 0:
            raise ValueError("Invalid SMILES input")

        # 1) Validate first (catch RDKit issues)
        try:
            logger.info("[predict] Validating SMILES: %s", smiles)
            is_valid, _mol, note = validate_smiles(smiles)
        except Exception as e:
            self._last_error = f"validate_smiles error: {e}"
            logger.exception("[predict] validate_smiles crashed: %s", e)
            raise

        if not is_valid:
            return {
                "valid": False,
                "image_png_b64": None,
                "latent": [],
                "properties": {},
                "meta": {
                    "model": "LTVAE-real" if self._is_real else "LTVAE-mock",
                    "reason": "Invalid SMILES",
                    "validator_note": note,
                    "load_error": self._last_error,
                },
            }

        # 2) Ensure model + run inference
        logger.info("[predict] Calling _ensure_loaded()")
        self._ensure_loaded()

        t0 = time.time()
        try:
            with torch.no_grad():
                logger.info("[predict] Inferring latent…")
                z = self._infer_latent(smiles)
                logger.info("[predict] Inferring properties…")
                props = self._infer_properties(z)
        except Exception as e:
            self._last_error = f"inference error: {e}"
            logger.exception("[predict] Inference crashed: %s", e)
            raise

        # 3) 2D image (don’t fail request if RDKit draw fails)
        try:
            img_b64 = smiles_png_base64(smiles)
        except Exception as e:
            img_b64 = None
            self._last_error = f"smiles_png_b64 error: {e}"
            logger.exception("[predict] RDKit image generation failed: %s", e)

        latency_ms = int((time.time() - t0) * 1000)
        meta = {
            "model": "LTVAE-real" if self._is_real else "LTVAE-mock",
            "latent_dim": int(z.shape[1]),
            "has_property_head": self._has_prop,
            "latency_ms": latency_ms,
            "validator_note": note,
            "image_generated": img_b64 is not None,
            "config_used": {
                "d_model": self._d_model, "latent_dim": self._latent_dim,
                "nhead": self._nhead, "enc_layers": self._enc_layers,
                "dec_layers": self._dec_layers, "ff_dim": self._ff_dim,
                "dropout": self._dropout, "max_len": self._max_len,
                "pad_idx": self._pad_idx, "sos_idx": self._sos_idx, "eos_idx": self._eos_idx
            },
            "load_error": self._last_error,
        }
        if self._has_prop:
            meta["prop_head"] = {
                "best_val_mse_stdspace": self._prop_meta.get("best_val_mse_stdspace", None),
                "metrics": self._prop_meta.get("val_metrics", {}),
                "props_order": self._props,
            }

        return {
            "valid": True,
            "image_png_b64": img_b64,
            "latent": [float(x) for x in z.squeeze(0).tolist()],
            "properties": props,
            "meta": meta,
        }