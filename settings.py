from pathlib import Path
import os.path 

root = Path(__file__).resolve().parent  # top-level directory
DATA_PATH = root / 'datasets/'  # datasets: change it if you want to save data elsewhere
EVAL_PATH = root / 'logs/'  # evaluation results

