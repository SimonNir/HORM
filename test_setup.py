#!/usr/bin/env python3
"""
Quick setup verification script for ESEN on HORM.

This script checks that all dependencies, data, and checkpoints are available
before starting training.

Usage:
    python test_setup.py
"""

import sys
import os
from pathlib import Path

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check(condition, name, details=""):
    """Print check result with color coding."""
    if condition:
        print(f"{GREEN}✓{RESET} {name}")
        if details:
            print(f"  → {details}")
        return True
    else:
        print(f"{RED}✗{RESET} {name}")
        if details:
            print(f"  → {details}")
        return False

def warn(message):
    """Print warning message."""
    print(f"{YELLOW}⚠{RESET} {message}")

def info(message):
    """Print info message."""
    print(f"{BLUE}ℹ{RESET} {message}")

def main():
    print("="*70)
    print("ESEN on HORM - Setup Verification")
    print("="*70)
    print()
    
    all_checks_passed = True
    
    # Check Python version
    print(f"{BLUE}[1] Checking Python Environment{RESET}")
    python_version = sys.version_info
    all_checks_passed &= check(
        python_version >= (3, 8),
        "Python version",
        f"Using Python {python_version.major}.{python_version.minor}.{python_version.micro}"
    )
    print()
    
    # Check dependencies
    print(f"{BLUE}[2] Checking Dependencies{RESET}")
    
    try:
        import torch
        all_checks_passed &= check(True, "PyTorch", f"v{torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            warn("CUDA not available - training will use CPU (slow)")
    except ImportError:
        all_checks_passed &= check(False, "PyTorch", "Not installed - run: pip install torch")
    
    try:
        import pytorch_lightning as pl
        all_checks_passed &= check(True, "PyTorch Lightning", f"v{pl.__version__}")
    except ImportError:
        all_checks_passed &= check(False, "PyTorch Lightning", "Not installed - run: pip install pytorch-lightning")
    
    try:
        import torch_geometric
        all_checks_passed &= check(True, "PyTorch Geometric", f"v{torch_geometric.__version__}")
    except ImportError:
        all_checks_passed &= check(False, "PyTorch Geometric", "Not installed - needed for data loading")
    
    try:
        from fairchem.core.models.equiformer_v2.so3 import SO3_Embedding
        all_checks_passed &= check(True, "fairchem-core", "eSEN model available")
    except ImportError:
        all_checks_passed &= check(False, "fairchem-core", "Not installed - run: pip install fairchem-core")
    
    try:
        import wandb
        check(True, "WandB (optional)", f"v{wandb.__version__}")
    except ImportError:
        warn("WandB not installed - use --no_wandb flag for training")
    
    print()
    
    # Check directory structure
    print(f"{BLUE}[3] Checking Directory Structure{RESET}")
    
    base_dir = Path(__file__).parent
    
    # Check data
    data_dir = base_dir / "data"
    all_checks_passed &= check(data_dir.exists(), "Data directory", str(data_dir))
    
    sample_data = data_dir / "sample_100.lmdb"
    all_checks_passed &= check(sample_data.exists(), "Sample dataset", str(sample_data))
    
    train_data = data_dir / "train.lmdb"
    if train_data.exists():
        info(f"Full training dataset found: {train_data}")
    else:
        warn("Full training dataset not found (optional for testing)")
        warn("Download from: https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/data")
    
    # Check checkpoints
    ckpt_dir = base_dir / "ckpt"
    all_checks_passed &= check(ckpt_dir.exists(), "Checkpoint directory", str(ckpt_dir))
    
    conserving_ckpt = ckpt_dir / "esen_sm_conserving_all.pt"
    all_checks_passed &= check(conserving_ckpt.exists(), "ESEN conserving checkpoint", str(conserving_ckpt))
    
    direct_ckpt = ckpt_dir / "esen_sm_direct_all.pt"
    all_checks_passed &= check(direct_ckpt.exists(), "ESEN direct checkpoint", str(direct_ckpt))
    
    # Check scripts
    train_script = base_dir / "train_esen_comparison.py"
    all_checks_passed &= check(train_script.exists(), "Training script", str(train_script))
    
    submit_script = base_dir / "submit_esen_comparison.sh"
    all_checks_passed &= check(submit_script.exists(), "Submission script", str(submit_script))
    
    print()
    
    # Check output directories
    print(f"{BLUE}[4] Checking Output Directories{RESET}")
    
    checkpoint_dir = base_dir / "checkpoint"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(exist_ok=True)
        info(f"Created checkpoint directory: {checkpoint_dir}")
    else:
        check(True, "Checkpoint output directory", str(checkpoint_dir))
    
    logs_dir = base_dir / "logs"
    if not logs_dir.exists():
        logs_dir.mkdir(exist_ok=True)
        info(f"Created logs directory: {logs_dir}")
    else:
        check(True, "Logs directory", str(logs_dir))
    
    print()
    
    # Check model integration
    print(f"{BLUE}[5] Checking Model Integration{RESET}")
    
    esen_module = base_dir / "nets" / "eSEN" / "esen_wrapper.py"
    all_checks_passed &= check(esen_module.exists(), "eSEN wrapper", str(esen_module))
    
    training_module = base_dir / "training_module.py"
    all_checks_passed &= check(training_module.exists(), "Training module", str(training_module))
    
    print()
    
    # Final summary
    print("="*70)
    if all_checks_passed:
        print(f"{GREEN}✓ All critical checks passed!{RESET}")
        print()
        print("You're ready to start training! Try:")
        print()
        print(f"  {BLUE}# Quick test (5 minutes){RESET}")
        print("  python train_esen_comparison.py --mode ef --data data/sample_100.lmdb --max_epochs 20 --no_wandb")
        print()
        print(f"  {BLUE}# Full comparison on sample data (1-2 hours){RESET}")
        print("  python train_esen_comparison.py --mode all --data data/sample_100.lmdb --max_epochs 200")
        print()
        print(f"  {BLUE}# Submit to cluster{RESET}")
        print("  sbatch submit_esen_comparison.sh")
    else:
        print(f"{RED}✗ Some checks failed!{RESET}")
        print()
        print("Please resolve the issues above before training.")
        print()
        print("Common fixes:")
        print("  • Install dependencies: pip install -r requirements_no_torch.txt")
        print("  • Install torch: pip install torch==2.2.1")
        print("  • Install fairchem: pip install fairchem-core")
        print("  • Download checkpoints from: https://huggingface.co/yhong55/HORM")
    
    print("="*70)
    print()
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
