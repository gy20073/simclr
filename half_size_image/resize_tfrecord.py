import tensorflow as tf
import sys, math, os, glob
import cv2
import numpy as np

# some constants begin
resize_ratio = 224.0 / 84.0
input_path = "/shared/yang/imagenet_original_tfdataset/5.0.0_original/*tfrecord*"
output_prefix = "/home/yang/code2/simclr_data/ilsvrc/imagenet2012/7.0.0/"

# some constants end

tf.enable_eager_execution()

# The following functions can be used to convert a value to a type compatible
# with tf.Example.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def half_size(image):
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, flags=cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, ( int(image.shape[1]/resize_ratio), int(image.shape[0]/resize_ratio) ))
    image = cv2.imencode('.jpeg', image)[1]
    return image.tobytes()

# Create a dictionary with features that may be relevant.
def image_example(file_name, image, label):
    feature = {
        'file_name': _bytes_feature(file_name),
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def convert_one_tfrecord(record_in, record_out):
    raw_image_dataset = tf.data.TFRecordDataset(record_in)

    # Create a dictionary describing the features.
    image_feature_description = {
        'file_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    with tf.io.TFRecordWriter(record_out) as writer:
        for image_features in parsed_image_dataset:
            image_raw = image_features['image'].numpy()
            image_raw = half_size(image_raw)

            tf_example = image_example(image_features['file_name'], image_raw, image_features['label'])
            writer.write(tf_example.SerializeToString())

def get_out_name(name_in, output_prefix):
    path, name = os.path.split(name_in)
    return os.path.join(output_prefix, name)

if __name__ == '__main__':
    nproc = int(sys.argv[1])
    pid = int(sys.argv[2])
    print(nproc, pid)

    input_names = sorted(glob.glob(input_path))
    each_len = int(math.ceil(len(input_names) * 1.0 / nproc))

    print(range(each_len * pid, min(each_len * (pid + 1), len(input_names))))

    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)

    for i in range(each_len * pid, min(each_len * (pid + 1), len(input_names))):
        iname = input_names[i]
        oname = get_out_name(iname, output_prefix)
        convert_one_tfrecord(iname, oname)
        print("done ", i)