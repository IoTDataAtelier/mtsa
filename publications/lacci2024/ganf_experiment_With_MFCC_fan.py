import os
import tensorflow as tf
import torch
import pandas as pd
import numpy as np 
import sys
import scipy.stats as st 
from sklearn.model_selection import KFold, cross_val_score
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

    batch_size_values = np.array([1024,512,256,128,64,32])
    learning_rate_values = np.array([1e-9])
    sampling_rate_sound = 16000

    column_names = [
        'batch_size' ,
        'epoch_size' ,
        'learning_rate' ,
        'sampling_rate' ,
        'AUC_ROCs',
        'Confidence_interval_AUC_ROC'
    ]

    experiment_dataframe = pd.DataFrame(columns=column_names)

    # paths = [
    #     '/data/MIMII/slider/id_00/',
    #     '/data/MIMII/slider/id_02/',
    #     '/data/MIMII/slider/id_04/',
    #     '/data/MIMII/slider/id_06/',
    # ]

    path = "/data/MIMII/fan/" 
    machine_ids = ['id_00', 'id_02', 'id_04', 'id_06']

    for machine_id in machine_ids:
        full_path = path+machine_id
        X_train, X_test, Y_train, Y_test = files_train_test_split(full_path)

        kf = KFold(n_splits=5)
        dataset_splits = list(enumerate(kf.split(X_train, Y_train)))

        for learning_rate in learning_rate_values:
            for batch_size in batch_size_values:
                print('\nlr= {}, batch= {}\n'.format(learning_rate, batch_size))
                scores = []
                for fold, (train_index, test_index) in dataset_splits:
                    print(fold + 1)

                    x_train_fold, y_train_fold = X_train[train_index], Y_train[train_index]

                    model_GANF = GANF(sampling_rate=sampling_rate_sound, mono= True, use_array2mfcc= True, isForWaveData= True)

                    model_GANF.fit(x_train_fold, y_train_fold, batch_size=int(batch_size), learning_rate=learning_rate)

                    auc = calculate_aucroc(model_GANF, X_test, Y_test)
                    scores.append(auc)
                    del model_GANF

                experiment_dataframe.loc[len(experiment_dataframe)] = {'batch_size': batch_size, 'epoch_size': 20, 'learning_rate': learning_rate, 'sampling_rate': sampling_rate_sound,'AUC_ROCs': str(scores)}
                experiment_dataframe.to_csv(csv_name +'_fold_'+str(fold+1)+'.csv', sep=',', encoding='utf-8')
        experiment_dataframe.to_csv(f'EXP_SOUND_MFCC_FAN_{machine_id}.csv', sep=',', encoding='utf-8')


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

with torch.cuda.device(0):
    if __name__ == '__main__':
        set_start_method('spawn')
        info('main line')
        p = Process(target=run_ganf_experiment)
        p.start()
        p.join()
