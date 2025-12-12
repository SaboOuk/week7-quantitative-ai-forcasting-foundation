import os
import subprocess
import sys


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def test_end_to_end_runs():
    # 1) generate data
    run([sys.executable, "data_generator.py"])
    assert os.path.exists("data/historical_volumes.csv")

    # 2) run forecast system
    run([sys.executable, "forecast_system.py"])
    assert os.path.exists("results/volume_analysis.png")
    assert os.path.exists("results/analysis_summary.txt")