torchrun --nproc_per_node 8 -m training.main_llava --batch-size=2 --lr=1e-5 --wd=0.1 --epochs=6 --workers=4 \
--model='openai/clip-vit-large-patch14-336' --warmup 1000 --dataset-type grid_distill  \
--train-data data/coco/annotations/instances_train2017.json \
--train-image-root="" \
--train-ceph-root="" \
--log-every-n-steps 50 \
--save-frequency 3 --lock-image-unlocked-groups 24 \
--name clipself_cc3m_6_save3_llava_openai_vitl14_24layers \
--alpha 0.95 --llava