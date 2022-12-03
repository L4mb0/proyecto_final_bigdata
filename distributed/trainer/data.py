import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from math import ceil
from trainer.utils import Mode
from trainer.utils import train_dir, test_dir

def _get_global_batch_size(strategy, batch_size):
    return strategy.num_replicas_in_sync * batch_size


def get_dataset(strategy, batch_size, mode):
    global_batch_size = _get_global_batch_size(strategy, batch_size)
    dataset = None
    if mode == Mode.TRAIN:
        train_datagen = ImageDataGenerator(brightness_range=(0.8,1.2), rotation_range=30, width_shift_range=0.2,
                                    height_shift_range=0.2, fill_mode='nearest', shear_range=0.2, zoom_range=0.3, rescale=1./255)

        train_generator = train_datagen.flow_from_directory(train_dir,target_size=(160,160),color_mode='rgb',
                                                    class_mode='binary',batch_size=batch_size,shuffle=True)
        
        dataset = tf.data.Dataset.from_generator(
            lambda: train_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, 160, 160, 3],[None,])
        )
    elif mode == Mode.VALID:
        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        valid_generator = valid_datagen.flow_from_directory(test_dir,target_size=(160,160),color_mode='rgb',
                                                    class_mode='binary',batch_size=batch_size)
        
        dataset = tf.data.Dataset.from_generator(
            lambda: valid_generator,
            output_types=(tf.float32, tf.float32),
            output_shapes = ([None, 160, 160, 3],[None,])
        )
    
        dataset = dataset.shuffle(10 * batch_size)
    
    dataset = dataset.repeat()
    dataset = dataset.batch(global_batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def get_steps_per_epoch(strategy, nr_of_examples, batch_size):
    return ceil(nr_of_examples / _get_global_batch_size(strategy, batch_size))