import os
from enum import Enum, unique
from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent.parent

dataset_dir = BASE_DIR / "dataset/"
train_dir = dataset_dir / 'train/'
test_dir = dataset_dir / 'test/'

BATCH_SIZE = 32

@unique
class Mode(Enum):
    TRAIN = 1
    VALID = 2

def _is_chief(task_type, task_id):
    '''Determines of machine is chief.'''
    return task_type == 'chief'


def _get_temp_dir(dirpath, task_id):
    '''Gets temporary directory for saving model.'''

    base_dirpath = 'workertemp_' + str(task_id)
    temp_dir = os.path.join(dirpath, base_dirpath)
    tf.io.gfile.makedirs(temp_dir)
    return temp_dir


def write_filepath(filepath, task_type, task_id):
    '''Gets filepath to save model.'''

    dirpath = os.path.dirname(filepath)
    base = os.path.basename(filepath)
    if not _is_chief(task_type, task_id):
        dirpath = _get_temp_dir(dirpath, task_id)
    return os.path.join(dirpath, base)