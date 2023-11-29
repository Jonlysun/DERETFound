IMAGENET_DIR=YOUR_OWN_PATH
python -m torch.distributed.launch --nproc_per_node=1 main_pretrain.py \
    --batch_size 224 \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 20 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path $IMAGENET_DIR \
    --task './DERETFound/' \
    --output_dir './DERETFound_log/' \
    # --resume ./mae_pretrain_vit_large.pth \

