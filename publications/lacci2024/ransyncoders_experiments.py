import os
import tensorflow as tf
import pandas as pd
import numpy as np 
import sys
from multiprocessing import Process

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from mtsa.metrics import calculate_aucroc
from mtsa.utils import files_train_test_split
from mtsa import RANSynCoders

def run_ransyncoders_experiment():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    batch_size_values = np.array([5000, 4000, 360, 180, 64, 32])
    learning_rate_values = np.array([1e-3,1e-9,1e-27,1,10])
    sampling_rate_sound = 16000

    path_input = '/data/MIMII/fan/id_00/'
    path_input1 = '/data/matheus-coelho/mtsa/examples/sample_data/machine_type_1/id_00'
    X_train, X_test, y_train, y_test = files_train_test_split(path_input)

    for learning_rate in learning_rate_values:
        for batch_size in batch_size_values:
            model_RANSynCoders = RANSynCoders(sampling_rate=sampling_rate_sound)
            model_RANSynCoders.fit(X_train, y_train, batch_size=batch_size, learning_rate=learning_rate)
            experiment_dataframe = model_RANSynCoders.get_config()
            auc = calculate_aucroc(model_RANSynCoders, X_test, y_test)
            experiment_dataframe.loc[len(experiment_dataframe)] = {'AUC_ROC':auc}
            experiment_dataframe.to_csv('exp_batch-{}_Lr-{}'.format(batch_size,learning_rate), sep='\t', encoding='utf-8')

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

if __name__ == '__main__':
    info('main line')
    p = Process(target=run_ransyncoders_experiment)
    p.start()
    p.join()