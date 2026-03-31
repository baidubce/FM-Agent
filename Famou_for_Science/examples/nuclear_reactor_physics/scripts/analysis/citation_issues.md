# Citation Verification Report

**Date**: 2026-03-14
**Verified by**: Background Researcher Agent
**Method**: 4-layer verification (Format → WebSearch → Author/Info → Content)

## Summary

- **Total entries checked**: 22
- **Verified and retained**: 21 (some with corrections)
- **Deleted**: 1 (ceolin2017fourier → replaced with zanette2018fourier)
- **Critical fixes applied**: 5 (wrong authors corrected)
- **Minor fixes applied**: 5 (missing authors/metadata added)

---

## Critical Issues Found and Fixed

### 1. `schiassi2022pinn` → RENAMED to `elhareef2022pinn`
- **Issue**: WRONG AUTHORS. The paper at DOI 10.1080/00295639.2022.2123211 is by **Elhareef, M.H. and Wu, Z.**, NOT Schiassi et al.
- **Root cause**: Confused two different 2022 PINN papers. Schiassi published a different paper on point kinetics equations in Annals of Nuclear Energy.
- **Fix**: Renamed key, corrected authors to Elhareef and Wu, updated pages to 601-622.
- **LaTeX impact**: Updated all `\cite{schiassi2022pinn}` → `\cite{elhareef2022pinn}` in introduction.tex and related_work.tex. Updated prose "Schiassi et al." → "Elhareef and Wu".

### 2. `kashi2013fourier` → RENAMED to `williams2013fourier`
- **Issue**: WRONG AUTHORS. The paper at DOI 10.1016/j.anucene.2013.06.020 is by **Williams, M.M.R., Hall, S.K., and Eaton, M.D.**, NOT Sherief et al.
- **Root cause**: Fabricated author names for a real paper.
- **Fix**: Renamed key, corrected authors.
- **LaTeX impact**: Updated `\cite{kashi2013fourier}` → `\cite{williams2013fourier}`, "Kashi et al." → "Williams et al."

### 3. `park2024deeponet` → RENAMED to `kobayashi2024deeponet`
- **Issue**: WRONG AUTHORS. The paper at DOI 10.1038/s41598-024-51984-x is by **Kobayashi, Kazuma and Alam, Syed Bahauddin**, NOT "Park, Jaewoong".
- **Root cause**: Fabricated first author name.
- **Fix**: Renamed key, corrected authors.
- **LaTeX impact**: Updated all `\cite{park2024deeponet}` → `\cite{kobayashi2024deeponet}`.

### 4. `tlpinn2023` — AUTHORS CORRECTED
- **Issue**: Wrong first author. Actual authors are **Prantikos, Konstantinos; Chatzidakis, Stylianos; Tsoukalas, Lefteri H.; Heifetz, Alexander** (NOT Radaideh).
- **Fix**: Corrected authors in BibTeX. Key unchanged.

### 5. `ceolin2017fourier` → REPLACED with `zanette2018fourier`
- **Issue**: DOI 10.1016/j.anucene.2017.04.035 could not be independently verified via WebSearch. The actual 2017 paper by this group was published in *Kerntechnik*, not *Annals of Nuclear Energy*.
- **Fix**: Replaced with the verified Zanette et al. 2018 paper (Annals of Nuclear Energy, vol 111, doi: 10.1016/j.anucene.2017.09.010) which covers the same FBPM method.
- **LaTeX impact**: Updated `\cite{ceolin2017fourier}` → `\cite{zanette2018fourier}`.

---

## Minor Issues Fixed

### 6. `parampinn2024` — Authors added
- **Before**: Author field was `{Atomic Energy Science and Technology}` (journal name, not authors!)
- **After**: `Xie, Yuchen and Ma, Yu and Wang, Yahui`
- Added DOI: 10.7538/yzk.2023.youxian.0765

### 7. `naspinn2025` — Authors and metadata added
- **Before**: No author field
- **After**: `Yu, Caiyang and Jiang, Yong and Chen, Qilong and Liu, Dong and Lyu, Jiancheng`
- Added volume=46, number=2, pages=119-126

### 8. `r2pinn2025` — Authors added
- **Before**: No author field
- **After**: `Zhang, Heng and He, Yunling and others`

### 9. `surrogate2026neuralop` → RENAMED to `sahadath2026neuralop`
- **Before**: Author field was `{arXiv preprint}` (placeholder!)
- **After**: `Sahadath, Md Hossain and Cheng, Qiyun and Pan, Shaowu and Ji, Wei`
- **LaTeX impact**: Updated `\cite{surrogate2026neuralop}` → `\cite{sahadath2026neuralop}`

### 10. `codeevolve2025` — Authors added
- **Before**: No author field
- **After**: `Assumpção, Henrique and Ferreira, Diego and Campos, Leandro and Murai, Fabricio`

---

## Verified Without Issues (11 entries)

| Key | Status | Notes |
|-----|--------|-------|
| `duderstadt1976nuclear` | ✅ Verified | Classic textbook, well-known |
| `bell1970nuclear` | ✅ Verified | Classic textbook, well-known |
| `lee2020nuclear` | ✅ Verified | Wiley, confirmed on publisher site |
| `kuridan2023neutron` | ✅ Verified | Springer, DOI confirmed |
| `osti1974fourier` | ✅ Verified | OSTI record confirmed |
| `momani2024twogroup` | ✅ Verified | Found on ResearchGate |
| `davierwalla1977fem` | ✅ Verified | Springer, DOI confirmed |
| `schmid1995fem` | ✅ Verified | Vieweg+Teubner publication |
| `raissi2019pinn` | ✅ Verified | JCP, DOI confirmed, 10000+ citations |
| `lu2021deeponet` | ✅ Verified | Nature MI, DOI confirmed |
| `li2021fno` | ✅ Verified | ICLR 2021, OpenReview confirmed |
| `funsearch2024` | ✅ Verified | Nature, DOI confirmed |
| `alphaevolve2025` | ✅ Verified | arXiv:2506.13131, DeepMind |

---

## Key Lessons

1. **Author fabrication was the most dangerous error**: 3 out of 22 entries had completely wrong first authors. This would have been caught by reviewers and could result in desk rejection.
2. **Missing authors**: 5 entries had placeholder or missing author fields — unacceptable for publication.
3. **DOI verification is essential**: The ceolin2017fourier DOI could not be confirmed; replaced with a verified alternative from the same research group.
4. **Principle applied**: "宁可少引用，不要保留有疑问的引用" — 1 entry replaced rather than kept with uncertain metadata.
