# SAR–SWIM Fusion

This repository implements an open-source workflow for fusing SAR and SWIM observations to improve global Stokes drift estimates.

## Citation

If you use this code in your research, please cite the following manuscript currently in preparation:

> **Zou Shihao, Hauser Danièle, Peureux Charles, Li Qing** (2026). *Fusing SAR and SWIM Observations to Improve the Global Stokes Drift Estimates*. Manuscript in preparation.

**BibTeX:**
```bibtex
@misc{zou2026fusing,
  title={Fusing SAR and SWIM Observations to Improve the Global Stokes Drift Estimates},
  author={Zou, Shihao and Hauser, Danièle and Peureux, Charles and Li, Qing},
  year={2026},
  note={Manuscript in preparation}
}
```

## Overview
- The main processing script `fuse_swiml2_wavetacl3.py` applies the SAR–SWIM fusion method to produce global Stokes drift estimates and aggregates daily results into a single NetCDF file.
- The `waveutils` module implements the fusion algorithm and related utilities:
    - `ambiguity_removal.sar_swim_fusion()` — performs the SAR–SWIM fusion.
    - `ambiguity_removal.remove_ambiguity_accord_wind()` — removes SWIM L2pDER directional ambiguities according to wind data.
