# /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/src/utils.py


from pathlib import Path


def get_figures_dir() -> Path:
    """
    Single source of truth for all experiment figures.
    """
    project_root = Path(__file__).resolve().parents[1]
    figures_dir = project_root / "experiments" / "simulation_experiment_run"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir
