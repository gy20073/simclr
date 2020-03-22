from absl import flags
import tensorflow.compat.v1 as tf
from data import get_preprocess_fn

FLAGS = flags.FLAGS

def build_input_fn_foreval(builder, is_training):
    def _input_fn(params):
        batch_size_cnn = params['batch_size']
        """Inner input function."""
        # TODO: here is the difference, we assume the linear eval phase does not have augmentation,
        #  while the original work did
        preprocess_fn_finetune = get_preprocess_fn(is_training=False, is_pretrain=False)
        num_classes = builder.info.features['label'].num_classes

        def map_fn(image, label):
            """Produces multiple transformations of the same batch."""
            image = preprocess_fn_finetune(image)
            return image, label, 1.0

        dataset = builder.as_dataset(
            split=FLAGS.train_split if is_training else FLAGS.eval_split,
            shuffle_files=False, as_supervised=True)

        dataset = dataset.map(map_fn,
                              num_parallel_calls=40)  # tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size_cnn, drop_remainder=False)
        #dataset = pad_to_batch(dataset, batch_size_cnn)

        images, labels, mask = tf.data.make_one_shot_iterator(dataset).get_next()
        #return images, {'labels': labels, 'mask': mask}
        return {'images': images, 'labels': labels}, {'mask': mask}

    return _input_fn
