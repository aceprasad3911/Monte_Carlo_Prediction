# /Users/ayushmaanprasad/Desktop/GitRepo/Monte_Carlo_Prediction/src/utils.py


import os

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_figures_dir() -> str:
    figures_dir = os.path.join(get_project_root(), "figures")
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir
