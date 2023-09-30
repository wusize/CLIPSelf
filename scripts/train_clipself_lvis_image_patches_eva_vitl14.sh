torchrun --nproc_per_node 8 -m training.main --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model EVA02-CLIP-L-14-336 --pretrained eva --warmup 1000  --zeroshot-frequency 1 --dataset-type grid_distill  \
--test-type coco_panoptic --train-data data/lvis_v1/annotations/lvis_v1_train.json \
--val-data data/coco/annotations/panoptic_val2017.json \
--embed-path metadata/coco_panoptic_clip_hand_craft_EVACLIP_ViTL14x336.npy --train-image-root data/lvis_v1 \
--val-image-root data/coco/val2017  --cache-dir checkpoints/EVA02_CLIP_L_336_psz14_s6B.pt --log-every-n-steps 50 \
--lock-image --save-frequency 6 --lock-image-unlocked-groups 24 --extract-type="v2" \
--name clipself_lvis_6_save6_test1_eva_vitl14_24layers --downsample-factor 14 --det-image-size 896 \
--alpha 0.95