#!/bin/bash

python main.py \
    --batch_size=2 \
    --sw_batch_size=2 \
    --logdir=candi_nnunet_fold0 \
    --optim_lr=5e-4 \
    --lrschedule=poly \
    --infer_overlap=0.5 \
    --save_checkpoint \
    --data_dir=dataset/Task912_CANDI \
    --json_list=dataset/Task912_CANDI/dataset.json \
    --roi_x=96 \
    --roi_y=112 \
    --roi_z=96 \
    --out_channels=28 \
    --workers=8 \
    --val_every=10 \
    --max_epochs=800 \
    --dataset=CANDI \
    --fold=0
