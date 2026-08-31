"""
Training Pipeline Orchestrator.
Runs the entire ML pipeline from data preparation to model registration.
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path

# Ensure parent directory is on sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run Training Pipeline")
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM training step for faster automated runs",
    )
    return parser.parse_args()

def run_script(script_path: str):
    """Run a python script as a subprocess and stream its output."""
    logger.info(f"========== RUNNING {script_path} ==========")
    try:
        # Use sys.executable to ensure we use the same Python interpreter (e.g. from the venv)
        cmd = [sys.executable, script_path]
        subprocess.run(cmd, check=True, cwd=str(project_root))
        logger.info(f"========== FINISHED {script_path} ==========\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"========== FAILED {script_path} (Exit Code: {e.returncode}) ==========")
        raise RuntimeError(f"Script {script_path} failed.") from e

def main():
    logger.info("--- Environment Variables Health Check ---")
    for var in config.REQUIRED_ENV_VARS:
        val = os.getenv(var)
        if val:
            logger.info(f"{var}: ✓ loaded")
        else:
            logger.warning(f"{var}: ✗ missing")
            
    args = parse_args()
    
    scripts = [
        "src/training/data_prep.py",
        "src/training/baseline_models.py",
    ]
    
    if not args.skip_lstm:
        scripts.append("src/training/deep_model.py")
    else:
        logger.info("Skipping deep_model.py due to --skip-lstm flag.")
        
    scripts.extend([
        "src/training/model_comparison.py",
        "src/training/model_registry.py"
    ])
    
    try:
        for script in scripts:
            run_script(script)
            
        logger.info("✅ Training pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
