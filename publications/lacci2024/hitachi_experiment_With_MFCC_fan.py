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

from mtsa.models.hitachi import Hitachi
from mtsa.metrics import calculate_aucroc
from mtsa.models.ganf import GANF
from mtsa.utils import files_train_test_split

def run_ganf_experiment():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

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

    path = "/data/MIMII/fan/" 
    machine_ids = ['id_00', 'id_02', 'id_04', 'id_06']

    for machine_id in machine_ids:
        full_path = path+machine_id
        X_train, X_test, Y_train, Y_test = files_train_test_split(full_path)

        kf = KFold(n_splits=5)
        dataset_splits = list(enumerate(kf.split(X_train, Y_train)))

        scores = []
        for fold, (train_index, test_index) in dataset_splits:
            print(fold + 1)

            x_train_fold, y_train_fold = X_train[train_index], Y_train[train_index]

            model_Hitachi= Hitachi()
            model_Hitachi.fit(x_train_fold, y_train_fold)

            auc = calculate_aucroc(model_Hitachi, X_test, Y_test)
            scores.append(auc)
            del model_Hitachi

        scores_mean = np.mean(scores)
        ci = st.t.interval(confidence=0.95, df=len(scores)-1, loc=scores_mean, scale=st.sem(scores))
        experiment_dataframe.loc[len(experiment_dataframe)] = {'batch_size': 512, 'epoch_size': 50, 'learning_rate': 0.001, 'sampling_rate': sampling_rate_sound,'AUC_ROCs': str(scores), 'Confidence_interval_AUC_ROC': str(ci)}
        experiment_dataframe.to_csv(f'EXP_HITACHI_SOUND_MFCC_FAN_{machine_id}_'+str(fold+1)+'Folds'+'.csv', sep=',', encoding='utf-8')

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
