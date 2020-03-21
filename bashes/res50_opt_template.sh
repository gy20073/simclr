# tunable parameters
id="res50_opt"
export CUDA_VISIBLE_DEVICES=2
is_train_phase=false

batch_size=128
resnet_depth=50
# tunable parameters ends

cd ..

DATA_DIR="/home/yang/code2/simclr_data/ilsvrc"
MODEL_DIR="/home/yang/code2/simclr_data/$id"
export TF_XLA_FLAGS=--tf_xla_auto_jit=2

shared_cmd="--dataset=imagenet2012 \
            --image_size=224 \
            --eval_split=validation \
            --data_dir=$DATA_DIR \
            \
            --resnet_depth=$resnet_depth \
            --width_multiplier=1.0 \
            \
            --optimizer=momentum \
            --train_batch_size=$batch_size\
            --weight_decay=1e-6 \
            --use_fp16=True \
            --train_summary_steps=600 \
            "
echo $shared_cmd

if $is_train_phase
then
  python run.py \
    --train_mode=pretrain \
    --train_epochs=500 \
    --learning_rate=0.3 \
    --temperature=0.1 \
    --model_dir=$MODEL_DIR \
    $shared_cmd
else
  python run.py \
    --mode=train_then_eval \
    --train_mode=finetune \
    --fine_tune_after_block=4 \
    --zero_init_logits_layer=True \
    --variable_schema='(?!(global_step|current_loss_scale|good_steps)|(?:.*/|^)Momentum|head)' \
    --learning_rate=0.1 \
    --train_epochs=90 \
    --warmup_epochs=0 \
    --model_dir=$MODEL_DIR"_ft" \
    --checkpoint=$MODEL_DIR \
    $shared_cmd
fi