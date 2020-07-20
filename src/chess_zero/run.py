"""
Main entry point for running from command line.
"""

import os
import sys
import multiprocessing as mp
import keras.backend.tensorflow_backend as tfback
import tensorflow as tf

#code is taken from here : https://github.com/keras-team/keras/issues/13684#issuecomment-595054461

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).
    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]
#experimental_list_devices is deprecated in tf 2.1

_PATH_ = os.path.dirname(os.path.dirname(__file__))


if _PATH_ not in sys.path:
    sys.path.append(_PATH_)

tfback._get_available_gpus = _get_available_gpus

if __name__ == "__main__":
    mp.set_start_method('spawn')
    sys.setrecursionlimit(10000)
    from chess_zero import manager
    manager.start()
