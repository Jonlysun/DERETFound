# DATASETS : ['DR_APTOS2019','DR_IDRID','DR_MESSIDOR2','Glaucoma_PAPILA','Glaucoma_Glaucoma_Fundus','Glaucoma_ORIGA','AMD_AREDS','Multi_Retina', 'Multi_JSIEC']
DATASET='DR_APTOS2019'
python -m torch.distributed.launch --nproc_per_node=1 --master_port=48797 main_finetune.py \
    --batch_size 32 \
    --world_size 1 \
    --model vit_large_patch16 \
    --epochs 50 \
    --blr 5e-3 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.2 \
    --root YOUR_OWN_PATH \
    --task ./checkpoint/$DATASET/  \
    --dataset_name $DATASET \
    --finetune ./checkpoint/PreTraining/checkpoint-best.pth
