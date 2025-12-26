# eSEN Integration for HORM

This directory contains the eSEN model integration for the HORM benchmarking framework.

## Files

- **`esen_wrapper.py`**: Main wrapper class that loads eSEN checkpoints via fairchem's `FAIRChemCalculator` and provides a PyTorch-compatible interface
- **`data_converter.py`**: Utilities for converting between HORM's PyG Data format and fairchem's AtomicData format
- **`__init__.py`**: Module initialization

## Usage

### Zero-shot Evaluation

Evaluate pretrained eSEN on HORM dataset without fine-tuning:

```bash
cd /home/simonnir/esen_horm/HORM
python eval_esen.py --checkpoint ckpt/esen_sm_direct_all.pt --data data/sample_100.lmdb
```

This will compute energy, force, and Hessian MAE metrics and save results to a text file.

### Fine-tuning

Fine-tune eSEN on HORM data with E+F supervision:

```bash
python train_esen.py
```

Edit `train_esen.py` to:
- Change `checkpoint_path` to use different checkpoint
- Modify `version` variable for different experiment names
- Adjust hyperparameters (learning rate, batch size, etc.)

## Model Details

- **Architecture**: eSEN (Efficiently Scaled Equivariant Network)
- **Checkpoints**: 
  - `ckpt/esen_sm_direct_all.pt`: Direct energy prediction (smooth, autograd-based forces)
  - `ckpt/esen_sm_conserving_all.pt`: Energy-conserving variant
- **Force Computation**: Forces are computed via autograd (`-dE/dpos`) for smoothness
- **Hessian Support**: Hessians computed via second-order autograd for training

## Integration with HORM

The eSEN wrapper integrates seamlessly with HORM's existing PyTorch Lightning training framework:

1. Model is registered in `training_module.py` as `model_config['name'] = 'eSEN'`
2. Uses the same data loaders, loss functions, and training loop as other models
3. Supports E+F and E+F+H training via existing Hessian loss implementation

## Requirements

- `fairchem-core`: Install with `pip install fairchem-core`
- PyTorch, PyTorch Geometric, PyTorch Lightning (already in HORM requirements)
