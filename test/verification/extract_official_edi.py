"""Re-extract the official Phoenix MT-Editor EDI (B0940330_z.edi) Z values
for the one-night window 2026-06-18 11:00-23:00 UTC (DESIGN §24.12
parity ground truth). Manual section parser (mt_metadata EDI.read()
returns None for this file).

Run:  .venv/bin/python test/verification/extract_official_edi.py
Output: test/verification/official_night_edi.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("test/verification")
EDI = Path("examples/data/phoenix-mtu-test/B0940330_z.edi")
MU0 = 4e-7 * np.pi
EDI_TO_SI = 1e-6 * MU0 / 1e-9  # mV/km per nT -> V/m per A/m = 1/795.77


def _section(text: str, marker: str) -> np.ndarray:
    lines = text.split(marker, 1)[1].splitlines()
    nums = []
    for ln in lines:
        s = ln.strip()
        if s.startswith(">"):
            break
        if s.startswith("//"):  # wrap-count header line
            continue
        nums += [float(x) for x in s.split() if not x.startswith("//")]
    return np.array(nums)


def read_official_edi(path: Path) -> dict[str, np.ndarray]:
    """Parse >FREQ and the ROT=ZROT Z sections (SEG 1.0, //30 wrapping)."""
    text = path.read_text()
    freq = _section(text, ">FREQ")
    assert freq.size == 80, freq.size
    zxy = _section(text, ">ZXYR ROT=ZROT") + 1j * _section(
        text, ">ZXYI ROT=ZROT")
    zyx = _section(text, ">ZYXR ROT=ZROT") + 1j * _section(
        text, ">ZYXI ROT=ZROT")
    assert zxy.size == 80 and zyx.size == 80
    return {"frequency": freq, "zxy": zxy, "zyx": zyx}


if __name__ == "__main__":
    d = read_official_edi(EDI)
    per = 1.0 / d["frequency"]
    w = 2 * np.pi * d["frequency"]
    # SEG default units: mV/km per nT -> SI = Z_EDI x 1e-6 x mu0/1e-9
    # (= /795.77). Verified: SI rho reproduces our chain's rho and the
    # 180-deg exp(+iwt) flip accounts for the sign (DESIGN §24.12 parity).
    zxy_si = d["zxy"] * EDI_TO_SI
    zyx_si = d["zyx"] * EDI_TO_SI
    rows = []
    for j, p in enumerate(per):
        rows.append({
            "period_s": float(p),
            "zxy": [float(zxy_si[j].real), float(zxy_si[j].imag)],
            "zyx": [float(zyx_si[j].real), float(zyx_si[j].imag)],
        })
    (OUT / "official_night_edi.json").write_text(json.dumps(rows, indent=1))
    zxy, zyx = zxy_si, zyx_si
    print(f"wrote {OUT / 'official_night_edi.json'}: {len(rows)} periods, "
          f"{per.min():.4g}-{per.max():.4g} s")
    for name, z in (("xy", zxy), ("yx", zyx)):
        rho = np.abs(z) ** 2 / (MU0 * w)
        phi = np.degrees(np.angle(z))
        print(f"{name}: rho {rho.min():.3g}-{rho.max():.3g} ohm-m, "
              f"phi {phi.min():.1f}..{phi.max():.1f} deg "
              f"(zxy[0]={z[0]:.3g} ohm-m)")
