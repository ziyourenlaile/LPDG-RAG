nohup sh -c '
    CUDA_VISIBLE_DEVICES=3,4,5,6 accelerate launch \
        --multi_gpu \
        --num_processes=4 \
        --mixed_precision=bf16 \
        src/modelft/RM_train/train_pairwise_rm.py
' > /srv/nfs/home/njnu_zrq/RankCoT/src/modelft/RM_train/rm_train.log 2>&1 &