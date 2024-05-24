import os
import scipy.stats as st 
from sklearn.model_selection import KFold
import tensorflow as tf
import pandas as pd
import numpy as np 
import sys
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Process

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from mtsa.metrics import calculate_aucroc
from mtsa import RANSynCoders
from mtsa.models.ransyncorders import DataType

def files_train_test_split_ebay():

    X_test_path_ebay = '/data/matheus-coelho/ebay_data/test.csv' 
    X_train_path_ebay = '/data/matheus-coelho/ebay_data/train.csv'
    Y_test_path_ebay = '/data/matheus-coelho/ebay_data/test_label.csv'
    X_train = pd.read_csv(X_train_path_ebay, index_col=[0])
    X_test = pd.read_csv(X_test_path_ebay, index_col=[0])
    Y_test = pd.read_csv(Y_test_path_ebay, index_col=[0])
    Y_train = []
    X_train.fillna(0, inplace=True)

    Y_test = Y_test.values.reshape(-1)
    return X_train, X_test, Y_test, Y_train 


def run_ransyncoders_experiment():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    epochs_size_values = np.array([5, 10])
    batch_size_values = np.array([720, 360, 180, 64])
    learning_rate_values = np.array([1e-3,1e-6])
    
    #Train Data
    X_train, X_test, Y_test, Y_train = files_train_test_split_ebay()
    t_train = np.tile(X_train.index.values.reshape(-1,1), (1, X_train.shape[1]))
    
    ##Test Data
    t_test = np.tile(X_test.index.values.reshape(-1,1), (1, X_train.shape[1]))
    
    #Preprocessing
    xscaler = MinMaxScaler()
    x_train_scaled = xscaler.fit_transform(X_train.values)
    x_test_scaled = xscaler.transform(X_test.values)
    
    kf = KFold(n_splits=5)
    dataset_splits = list(enumerate(kf.split(x_train_scaled, t_train)))
    
    column_names = [
        'batch_size' ,
        'epoch_size' ,
        'learning_rate' ,
        'sampling_rate' ,
        'AUC_ROCs',
        'Confidence_interval_AUC_ROC'
    ]
    
    experiment_dataframe = pd.DataFrame(columns=column_names)
    
    for epochs in epochs_size_values:
        for learning_rate in learning_rate_values:
            for batch_size in batch_size_values:
                print('\nepochs_size= {}, lr= {}, batch= {}\n'.format(epochs, learning_rate, batch_size))
                scores = []
                for fold, (train_index, test_index) in dataset_splits:
                    print(fold + 1)

                    x_train_fold, t_train_fold = x_train_scaled[train_index], t_train[train_index]
                    
                    model_RANSynCoders = RANSynCoders(sampling_rate=1, data_type=DataType.DEFAULT)
                    
                    model_RANSynCoders.fit(x_train_fold, timestamps_matrix=t_train_fold, batch_size=batch_size, learning_rate=learning_rate, epochs=epochs)
                    
                    model_RANSynCoders.set_timestamps_matrix_to_predict(t_test)
                    
                    auc = calculate_aucroc(model_RANSynCoders, x_test_scaled, Y_test)
                    scores.append(auc)
                    del model_RANSynCoders
                scores_mean = np.mean(scores) 
                ci = st.t.interval(confidence=0.95, df=len(scores)-1, loc=scores_mean, scale=st.sem(scores))
                experiment_dataframe.loc[len(experiment_dataframe)] = {'batch_size': batch_size, 'epoch_size': epochs, 'learning_rate': learning_rate, 'sampling_rate': 1,'AUC_ROCs': str(scores), 'Confidence_interval_AUC_ROC': str(ci)}
                experiment_dataframe.to_csv('EXP_EBAY_RANSynCoders_MTSA_CV_'+'fold_'+str(fold)+'.csv', sep=',', encoding='utf-8')
                
    experiment_dataframe.to_csv('EXP_EBAY_RANSynCoders_MTSA_CV.csv', sep=',', encoding='utf-8')
                    
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