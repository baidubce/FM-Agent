from pathlib import Path


EXAMPLE_ROOT = Path(__file__).resolve().parents[2]
FAMOU_TASK_DIR = EXAMPLE_ROOT / "famou" / "task1"
BASELINES_DIR = EXAMPLE_ROOT / "baselines"
ANALYSIS_DIR = EXAMPLE_ROOT / "scripts" / "analysis"
PAPER_DIR = EXAMPLE_ROOT / "paper"
PAPER_FIG_DIR = PAPER_DIR / "figs"


def ensure_output_dirs() -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    (BASELINES_DIR / "results").mkdir(parents=True, exist_ok=True)
