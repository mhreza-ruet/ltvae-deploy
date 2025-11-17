import base64, io, re
from typing import Tuple, Optional

# Try RDKit (preferred)
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

_ALLOWED = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789=#()[]+-/@\\.")
_BRACKETS = {"(": ")", "[": "]"}

def _quick_sanity(smiles: str) -> bool:
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    # allowed chars check
    if any(ch not in _ALLOWED for ch in smiles):
        return False
    # bracket balance
    stack = []
    for ch in smiles:
        if ch in _BRACKETS:
            stack.append(ch)
        elif ch in _BRACKETS.values():
            if not stack: return False
            op = stack.pop()
            if _BRACKETS[op] != ch: return False
    return len(stack) == 0

def validate_smiles(smiles: str) -> Tuple[bool, Optional["Chem.Mol"], str]:
    """
    Returns (is_valid, mol_or_None, note).
    If RDKit available, uses Chem.MolFromSmiles + sanitize for truth.
    Else falls back to quick sanity checks (best-effort).
    """
    if RDKit_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False, None, "RDKit: parse failed"
            Chem.SanitizeMol(mol)
            return True, mol, "RDKit: valid"
        except Exception as e:
            return False, None, f"RDKit error: {e}"
    else:
        ok = _quick_sanity(smiles)
        return ok, None, "RDKit not installed; quick validation" if ok else "Failed quick validation"

def smiles_png_base64(smiles: str, size=(320, 240)) -> Optional[str]:
    """
    Renders SMILES to a base64 PNG if RDKit is available; otherwise returns None.
    """
    if not RDKit_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=size, legends=[smiles])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")