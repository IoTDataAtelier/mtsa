import os
import tensorflow as tf
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
from mtsa.utils import files_train_test_split
from mtsa import RANSynCoders
from mtsa.models.ransyncorders import DataType

def run_ransyncoders_experiment():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    column_names = [
        'batch_size' ,
        'epoch_size' ,
        'learning_rate' ,
        'sampling_rate' ,
        'AUC_ROCs',
        'Confidence_interval_AUC_ROC'
    ]

    epochs_size_values = np.array([10, 15])
    learning_rate_values = np.array([1e-3, 1e-6])
    batch_size_values = np.array([720, 360, 180, 90, 45])
    machine_ids = ['id_00', 'id_02', 'id_04', 'id_06']
    sampling_rate_sound = 16000

    for machine_id in machine_ids:
        
        path_input = '/data/MIMII/fan/'
        path_input = os.path.join(path_input, machine_id)
        X_train, X_test, y_train, y_test = files_train_test_split(path_input)
    
        kf = KFold(n_splits=5)
        dataset_splits = list(enumerate(kf.split(X_train)))
        experiment_dataframe = pd.DataFrame(columns=column_names)
        
        for learning_rate in learning_rate_values:
            for epochs in epochs_size_values:
                for batch_size in batch_size_values:
                    print('\nepochs_size= {}, batch= {}\n'.format(epochs, batch_size))
                    scores = []
                    for fold, (train_index, test_index) in dataset_splits:
                        print(fold + 1)

                        x_train_fold = X_train[train_index]

                        model_RANSynCoders = RANSynCoders(sampling_rate=sampling_rate_sound, data_type=DataType.MFCC, mono=True)
                        
                        model_RANSynCoders.fit(x_train_fold, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
                        
                        auc = calculate_aucroc(model_RANSynCoders, X_test, y_test)
                        scores.append(auc)
                        del model_RANSynCoders
                    scores_mean = np.mean(scores)
                    ci = st.t.interval(confidence=0.95, df=len(scores)-1, loc=scores_mean, scale=st.sem(scores))
                    experiment_dataframe.loc[len(experiment_dataframe)] = {'batch_size': batch_size, 
                                                                           'epoch_size': epochs, 
                                                                           'learning_rate': str(learning_rate), 
                                                                           'sampling_rate': sampling_rate_sound,
                                                                           'AUC_ROCs': str(scores), 
                                                                           'Confidence_interval_AUC_ROC': str(ci)}
                    experiment_dataframe.to_csv(f'EXP_MFCC_FAN-{machine_id}_MTSA_CV_'+'fold_'+str(fold)+'.csv', sep=',', encoding='utf-8')
                    
        experiment_dataframe.to_csv(f'EXP_MFCC_FAN-{machine_id}_MTSA_CV.csv', sep=',', encoding='utf-8')
        
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