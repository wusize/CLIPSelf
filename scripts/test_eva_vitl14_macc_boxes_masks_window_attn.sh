NAME=$1
CHECKPOINT=$2
torchrun --nproc_per_node 8 -m training.main --batch-size=1 \
--model EVA02-CLIP-L-14-336 --pretrained eva --test-type coco_panoptic --train-data="" \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy \
--val-image-root data/coco/val2017  --cache-dir $CHECKPOINT --extract-type="v2" \
--name $NAME --downsample-factor 14 --det-image-size 896 \
--window-size 16 --window-attention