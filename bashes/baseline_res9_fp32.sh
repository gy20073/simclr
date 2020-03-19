DATA_DIR="/home/yang/code2/simclr_data/ilsvrc"
MODEL_DIR="/home/yang/code2/simclr_data/res9"

cd ..

export CUDA_VISIBLE_DEVICES=1

python run.py \
  --train_mode=pretrain \
  --train_batch_size=256 \
  --train_epochs=100 \
  --learning_rate=0.3 \
  --weight_decay=1e-6 \
  --temperature=0.1 \
  --dataset=imagenet2012 \
  --image_size=224 \
  --eval_split=validation \
  --data_dir=$DATA_DIR \
  --model_dir=$MODEL_DIR \
  --use_tpu=False \
  --train_summary_steps=0 \
  --backbone="resnet" \
  --width_multiplier=1.0 \
  --resnet_depth=9 \
  --global_bn=False