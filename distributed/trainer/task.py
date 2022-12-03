import tensorflow as tf
import argparse
import json
import os

from trainer.model import create_model

from trainer.utils import BATCH_SIZE
from trainer.utils import write_filepath
from trainer.utils import Mode
from trainer.utils import train_dir, test_dir

from trainer.data import get_dataset
from trainer.data import get_steps_per_epoch

def get_args():
    '''Parses args.'''

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--epochs',
        required=True,
        type=int,
        help='number training epochs')
    
    parser.add_argument(
        '--job-dir',
        required=True,
        type=str,
        help='bucket to save model')
    
    args = parser.parse_args()
    return args

def get_size_dataset(dir):
    categories = ['real','spoof']
    for category in categories:
        path = os.path.join(dir,category)
        if category == 'real':
            r1 = len(os.listdir(path))
        else:
            s1 = len(os.listdir(path))
            
    return r1 + s1

def main():
    args = get_args()
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    with strategy.scope():
        model = create_model()

    train_ds = get_dataset(strategy, BATCH_SIZE, Mode.TRAIN)
    valid_ds = get_dataset(strategy, BATCH_SIZE, Mode.VALID)
    
    steps_per_epoch = get_steps_per_epoch(
        strategy, 
        get_size_dataset(train_dir),
        BATCH_SIZE 
    )
    
    validation_steps = get_steps_per_epoch(
        strategy,
        get_size_dataset(test_dir),
        BATCH_SIZE,
    )
    
    model.fit(
        train_ds,
        steps_per_epoch = steps_per_epoch,
        validation_data = valid_ds, 
        validation_steps = validation_steps,
        epochs = args.epochs)

    # Determine type and task of the machine from 
    # the strategy cluster resolver
    task_type, task_id = (strategy.cluster_resolver.task_type,
                        strategy.cluster_resolver.task_id)

    # Based on the type and task, write to the desired model path 
    write_model_path = write_filepath(args.job_dir, task_type, task_id)
    model.save(write_model_path)

if __name__ == "__main__":
    main()