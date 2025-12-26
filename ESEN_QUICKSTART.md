# eSEN Integration - Quick Start Guide

## What's Been Set Up

I've integrated eSEN into your HORM framework. Here's what you can do now:

## 1. Zero-shot Evaluation (Start Here!)

Test the pretrained eSEN model on your 100-sample dataset:

```bash
cd /home/simonnir/esen_horm/HORM
python eval_esen.py --checkpoint ckpt/esen_sm_direct_all.pt --data data/sample_100.lmdb
```

This will:
- Load the pretrained eSEN checkpoint
- Evaluate on your HORM data
- Compute E, F, H, eigenvalue, and asymmetry MAE
- Save results to `results_esen_esen_sm_direct_all.txt`

You can also test the conserving checkpoint:
```bash
python eval_esen.py --checkpoint ckpt/esen_sm_conserving_all.pt --data data/sample_100.lmdb
```

## 2. Fine-tuning (After Zero-shot)

Once you've verified zero-shot works, fine-tune eSEN:

```bash
python train_esen.py
```

This uses the same PyTorch Lightning framework as your EquiV2 training.

### Customization

Edit `train_esen.py` to:
- Change checkpoint: `checkpoint_path="ckpt/esen_sm_conserving_all.pt"`
- Adjust learning rate: `lr=5e-5` (lower for fine-tuning)
- Change batch size: `bz=4`
- Modify loss weights in `training_module.py` line 336

## 3. File Structure

```
HORM/
├── nets/eSEN/              # eSEN module (NEW)
│   ├── esen_wrapper.py     # Model wrapper
│   ├── data_converter.py   # Format conversion
│   └── README.md           # Documentation
├── eval_esen.py            # Zero-shot evaluation (NEW)
├── train_esen.py           # Fine-tuning script (NEW)
├── training_module.py      # Updated with eSEN support
└── ckpt/
    ├── esen_sm_direct_all.pt
    └── esen_sm_conserving_all.pt
```

## 4. Next Steps

1. **Run zero-shot eval** on both checkpoints
2. **Compare results** with your EquiV2 baselines
3. **Fine-tune** with E+F (current setup)
4. **Add E+F+H** by adjusting loss weights in `training_module.py`
5. **Deploy to cluster** once validated locally

## Notes

- eSEN uses `forward_autograd()` for smooth forces (forces via autograd from energy)
- Hessians are computed using your existing `get_force_jac_loss()` method
- The wrapper handles all format conversions automatically
- Training uses your existing WandB logging and checkpointing

## Troubleshooting

If you get import errors:
```bash
pip install fairchem-core
```

If CUDA out of memory:
- Reduce batch size in `train_esen.py`: `bz=1`
- Or use CPU: `device="cpu"` in model_config
