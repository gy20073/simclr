export CUDA_VISIBLE_DEVICES=""

for i in {0..39}
do
  python resize_tfrecord.py 40 $i &
done