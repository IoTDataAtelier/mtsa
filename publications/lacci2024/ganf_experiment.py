import os
import tensorflow as tf
import torch
import pandas as pd
import numpy as np 
import sys
from multiprocessing import Process, set_start_method

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from mtsa.metrics import calculate_aucroc
from mtsa.models.ganf import GANF
from mtsa.utils import files_train_test_split

def run_ganf_experiment():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    batch_size_values = np.array([128,64,32,16])
    learning_rate_values = np.array([1e-3,1e-27,1e-9,1,10])
    sampling_rate_sound = 16000

    column_names = [
        'batch_size' ,
        'epoch_size' ,
        'current_epoch' ,
        'epoch_NaN' ,
        'learning_rate' ,
        'sampling_rate' ,
        'A_hat' ,
        'A_hat_NaN',
        'Adjacent_Matrix' ,
        'Adjacent_Matrix_NaN',
        'h' ,
        'loss_train' ,
        'loss_train_mean' ,
        'loss_best',
        'alpha',
        'rho',
        'total_loss',
        'AUC_ROC'
    ]

    experiment_dataframe = pd.DataFrame(columns=column_names)

    path_input = '/data/MIMII/fan/id_00/'
    #path_input_1 = '/data/cristofer/mtsa/examples/sample_data/machine_type_1/id_00'
    X_train, X_test, y_train, y_test = files_train_test_split(path_input)

    for batch_size in batch_size_values:
        for learning_rate in learning_rate_values:
            model_GANF = GANF(sampling_rate=sampling_rate_sound)
            model_GANF.fit(X_train, y_train, batch_size=batch_size, learning_rate=learning_rate, debug_dataframe=experiment_dataframe)
            experiment_dataframe['sampling_rate'] = experiment_dataframe['sampling_rate'].fillna(sampling_rate_sound)
            auc = calculate_aucroc(model_GANF, X_test, y_test)
            experiment_dataframe.loc[len(experiment_dataframe)] = {'AUC_ROC':auc}
            learning_rate_str = str(learning_rate).replace('.', '_')
            experiment_dataframe.to_csv('exp_batch-{}_Lr-{}_GANF_MTSA.csv'.format(batch_size,learning_rate_str), sep=',', encoding='utf-8')
            experiment_dataframe = pd.DataFrame(columns=column_names)

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

with torch.cuda.device(1):
    if __name__ == '__main__':
        set_start_method('spawn')
        info('main line')
        p = Process(target=run_ganf_experiment)
        p.start()
        p.join()