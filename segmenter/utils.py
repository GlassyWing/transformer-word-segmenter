import os
import tensorflow as tf


def get_session(gpu_fraction=0.6):
    '''''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads, allow_soft_placement=True))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
