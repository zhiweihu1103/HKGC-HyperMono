export LOG_PATH=./logs/wikipeople.out
export SAVE_DIR_NAME=wikipeople
export DATASET=wikipeople
export CUDA=5

export LABEL_SMOOTH=0.8
export BETA_WEIGHT=4.0
export HIDDEN_SIZE=400
export ABLATION_MODE=normal
export ENT_LOSS_WEIGHT=1.0
export INIT_WEIGHT=0.02

nohup python -u run.py \
   --device cuda:$CUDA \
   --dataset $DATASET \
   --train_mode without_valid \
   --save_dir_name $SAVE_DIR_NAME \
   --kge_label_smoothing $LABEL_SMOOTH \
   --beta_weight $BETA_WEIGHT \
   --hidden_size $HIDDEN_SIZE \
   --ablation_mode $ABLATION_MODE \
   --ent_neighbor_loss_weight $ENT_LOSS_WEIGHT \
   --initializer_range $INIT_WEIGHT \
   > $LOG_PATH 2>&1 &  