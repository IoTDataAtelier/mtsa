import os
import torch
import pandas as pd
import numpy as np 
import sys
import scipy.stats as st 
from sklearn.model_selection import KFold, cross_val_score
from multiprocessing import Process, set_start_method
import time
import tensorflow as tf

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from mtsa.metrics import calculate_aucroc
from mtsa.models.ransyncorders import RANSynCoders
from mtsa.utils import files_train_test_split

def run_RANSynCoders_experiment():
    
    machine_type = "fan"
    train_ids = ["id_02"]
    test_id = "id_00"
    model_name = 'RANSynCoders'
    
    save_dir = os.path.join(os.getcwd(), f'experiment_ood_{model_name}_only1Id', machine_type.upper())
    os.makedirs(save_dir, exist_ok=True)
    
    X_train, y_train = [], []
    for machine_id in train_ids:
        data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, machine_id)
        X_data, _, y_data, _ = files_train_test_split(data_path)  
        X_train.extend(X_data)
        y_train.extend(y_data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, test_id)
        
    _, X_test, _, y_test = files_train_test_split(data_path)

    #batch_size_values = np.array([512, 256, 128, 64, 32])
    learning_rate_values = np.array([1e-3, 1e-6])
    
    batch_size_values = np.array([720, 360, 180, 90, 45])

    #best_batch_size_value_valve = np.array([90])
    #best_learning_rate_value_valve = np.array([1e-3])
    
    #best_batch_size_value_pump = np.array([720])
    #best_learning_rate_value_pump = np.array([1e-6])
    
    #best_batch_size_value_fan = np.array([720])
    #best_learning_rate_value_fan = np.array([1e-6])
     
    #best_batch_size_value_slider = np.array([45])
    #best_learning_rate_value_slider = np.array([1e-6])
    
    kf = KFold(n_splits=5)
    dataset_splits = list(enumerate(kf.split(X_train, y_train)))
    
    for learning_rate in learning_rate_values:
            for batch_size in batch_size_values:
                
                result = []  
                print('---' * 20)
                print(f'Running for learning_rate={learning_rate}, batch_size={batch_size}')
                print(f'KFold\t|AUC\t|')

                # Validação cruzada K-Fold
                for i, (train_index, val_index) in dataset_splits:
                    # Registro do tempo de início do experimento
                    start = time.time() / 60.0

                    # Instancia o modelo RANSynCoders com os parâmetros atuais
                    model_RANSynCoders = RANSynCoders(is_acoustic_data=True, mono=True, normal_classifier=1, abnormal_classifier=0, synchronize=True)

                    # Seleção das amostras de treino para o fold atual
                    X_train_KFold = X_train[train_index]
                    y_train_fold = y_train[train_index]

                    # Treinamento do modelo com os dados de treino do fold
                    model_RANSynCoders.fit(X_train_KFold, y_train_fold)

                    # Predição no conjunto de teste
                    auc = calculate_aucroc(model_RANSynCoders, X_test, y_test)

                    # Registro do tempo de término do experimento
                    end = time.time() / 60.0

                    # Cálculo da duração da execução
                    execution_duration = end - start
                    
                    # Armazena os resultados de AUC e os parâmetros do modelo para cada fold
                    result.append([model_name, i, batch_size, learning_rate, execution_duration, auc])  
                    print(f'K({i}):\t|{auc:0.5f}\t|')

                # Salvar resultados em CSV
                experiment_name = f'{machine_type}_OOD_train-{train_ids}_test-{test_id}_batch_size-{batch_size}_learning_rate-{learning_rate}'
                csv_filename = os.path.join(save_dir, f'{experiment_name}.csv')

                # Cria DataFrame com os resultados e salva
                df_results = pd.DataFrame(result, columns=['Model', 'Fold', 'batch_size', 'learning_rate', 'Execution_Duration', 'AUC_ROC'])
                df_results.to_csv(csv_filename, sep=',', encoding='utf-8', index=False)

                print(f'Results saved to {csv_filename}')
                print('---' * 20)
    
    
def info(title):
    print(title)
    print("module name:", __name__)
    print("parent process:", os.getppid())
    print("process id:", os.getpid())

with tf.device('/gpu:1'):
    if __name__ == "__main__":
        set_start_method("spawn")
        info("main line")
        p = Process(target=run_RANSynCoders_experiment)
        p.start()
        p.join()