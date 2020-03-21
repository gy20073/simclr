export CUDA_VISIBLE_DEVICES=""

for i in {0..15}
do
  python resize_tfrecord.py 16 $i &
done