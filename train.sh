python train.py \
  --device=cpu \
  --checkpoint=ckpt/sam_med3d_turbo.pth \
  --batch_size=1 \
  --num_epochs=2 \
  --allow_partial_weight \
  --accumulation_steps=4 \
  --num_workers=8 \
  --freeze_encoder