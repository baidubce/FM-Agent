# Case data release

The Git repository contains all 49 case protocols and evaluators. Large inputs under `case/*/data/` and `case/*/tests/data/` are stored in GitHub Release archives to avoid multi-gigabyte Git history and GitHub's per-file Git limit.

## Restore available data

```bash
python scripts/fetch_data.py --list
python scripts/fetch_data.py
python scripts/fetch_data.py --check
```

The release has two logical archives:

| Archive | Contents |
| --- | --- |
| `fde-bench-data-case001.tar.gz` | Aerodynamic forecasting, case 001; **2,645,224,452 bytes** compressed, uploaded as two parts because it exceeds GitHub's 2 GiB asset limit |
| `fde-bench-data-cases002-049.tar.gz` | External inputs for cases 002–049 (case 040 is tracked in Git); **253,384,892 bytes** compressed |

The case 001 archive is uploaded as two parts: `fde-bench-data-case001.tar.gz.part01` (1,610,612,736 bytes) and `fde-bench-data-case001.tar.gz.part02` (1,034,611,716 bytes). An archive that exceeds GitHub's per-asset limit is uploaded as numbered parts. The installer downloads and verifies each part, rejoins them, checks the complete archive, and removes the cached parts after successful joining. It then verifies every extracted file. `data-manifest.json` records the exact filenames, order, byte counts, and SHA-256 checksums.

Allow approximately 12 GB of free disk space during downloading, joining, and extraction. A cache of the complete archives remains in `.cache/fde-bench-data/` and can be removed after installation. Dataset contents and download caches are excluded from Git.

For files downloaded separately, keep their release filenames and pass them to the installer:

```bash
python scripts/fetch_data.py \
  --archive /path/to/fde-bench-data-case001.tar.gz \
  --archive /path/to/fde-bench-data-cases002-049.tar.gz
```

If case 001 is split into parts, pass each part with its own `--archive` argument instead of its complete archive. The installer can combine local files with downloads for any remaining assets.

## Case 040

`040_frailty_cohort_analysis` uses a small synthetic cohort (four CSV files, about 320 KB). Its inputs, held-out outcomes, and reference output are tracked directly in Git under `case/040_frailty_cohort_analysis/` and are not part of the release archives. All **49 cases** have complete data.

## What changed for distribution

Business values and task definitions were not altered during packaging. Three spreadsheets had document-author metadata removed; worksheet payloads and other workbook entries remain unchanged. Released file checksums refer to those cleaned workbooks. The original local source directory is not modified by this packaging step.

Run outputs, credentials, Python caches, and dependency installations are not included. Case data are licensed under [CC BY-NC-ND 4.0](LICENSE-DATA); see [LICENSE_STATUS.md](LICENSE_STATUS.md) for scope and third-party exceptions.
