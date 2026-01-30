# src/utils.py


from pathlib import Path


def get_figures_dir(*subdirs):
    project_root = Path(__file__).resolve().parents[1]
    path = project_root / "experiments" / "simulation_experiment_run"
    for s in subdirs:
        path /= s
    path.mkdir(parents=True, exist_ok=True)
    return path
