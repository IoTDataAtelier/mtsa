import os
import torch
import pandas as pd
import numpy as np 
import sys
import scipy.stats as st 
from sklearn.model_selection import KFold
from multiprocessing import Process, set_start_method


module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from mtsa.metrics import calculate_aucroc
from mtsa.models.gacvae import GACVAE
from mtsa.utils import files_train_test_split

def run_gacvae_experiment():
    batch_size_values = np.array([512])
    learning_rate_values = np.array([1e-6])
    sampling_rate_sound = 16000

    column_names = [
        'batch_size' ,
        'epoch_size' ,
        'learning_rate' ,
        'sampling_rate' ,
        'AUC_ROCs',
    ]
    
    machine_names = ["fan", "valve", "pump", "slider"] 

    experiment_dataframe = pd.DataFrame(columns=column_names)

    path_input = "/data/MIMII/"
    id = "/id_00/"

    X_train, X_test, Y_train, Y_test = files_train_test_split(path_input+machine_names[3]+id)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    dataset_splits = list(enumerate(kf.split(X_train, Y_train)))

    for machine_name in machine_names:
        for learning_rate in learning_rate_values:
            for batch_size in batch_size_values:
                print('\nlr= {}, batch= {}\n'.format(learning_rate, batch_size))
                scores = []
                for fold, (train_index, test_index) in dataset_splits:
                    print(fold + 1)

                    x_train_fold, y_train_fold = X_train[train_index], Y_train[train_index]

                    model_gacvae = GACVAE(sampling_rate=sampling_rate_sound, device=1, isForWaveData=True, use_array2mfcc= True)
                    model_gacvae.fit(x_train_fold, y_train_fold, batch_size=int(batch_size), learning_rate=learning_rate, epochs=1, max_iteraction=1)

                    auc = calculate_aucroc(model_gacvae, X_test, Y_test)
                    scores.append(auc)

                    del model_gacvae

                experiment_dataframe.loc[len(experiment_dataframe)] = {'batch_size': batch_size, 'epoch_size': 20, 'learning_rate': learning_rate, 'sampling_rate': sampling_rate_sound,'AUC_ROCs': str(scores)}

        experiment_dataframe.to_csv('exp_GACVAE_'+ machine_name + '.csv', sep=',', encoding='utf-8')


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

with torch.cuda.device(1):
    if __name__ == '__main__':
        set_start_method('spawn')
        info('main line')
        p = Process(target=run_gacvae_experiment)
        p.start()
        p.join()
