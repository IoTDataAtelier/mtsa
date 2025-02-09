import os
import torch
import pandas as pd
import numpy as np 
import sys
import scipy.stats as st 
from sklearn.model_selection import KFold, cross_val_score
from multiprocessing import Process, set_start_method
import time

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

from mtsa.metrics import calculate_aucroc
from mtsa.models.ganf import GANF
from mtsa.utils import files_train_test_split

def run_ganf_experiment():
    
    machine_type = "pump"
    train_ids = ["id_02"]
    test_id = "id_00"
    model_name = 'GANF'
    
    save_dir = os.path.join(os.getcwd(), f'experiment_ood_{model_name}_bestHyperparametersTestPump5', machine_type.upper())
    os.makedirs(save_dir, exist_ok=True)
    
    X_train, y_train = [], []
    for machine_id in train_ids:
        data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, machine_id)
        X_data, _, y_data, _ = files_train_test_split(data_path)  
        X_train.extend(X_data)
        y_train.extend(y_data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Loaded {len(X_train)} test samples for machine_type='{machine_type}'")
    
    data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, test_id)
        
    _, X_test, _, y_test = files_train_test_split(data_path)
    
    print(f"Loaded {len(X_test)} test samples for machine_type='{machine_type}'")

    #batch_size_values = np.array([512, 256, 128, 64, 32])
    #learning_rate_values = np.array([1e-3, 1e-6])
    
    #best_batch_size_value_valve = np.array([32])
    #best_learning_rate_value_valve = np.array([1e-3])
    
    best_batch_size_value_pump = np.array([32])
    best_learning_rate_value_pump = np.array([1e-6])
    
    #best_batch_size_value_fan = np.array([128])
    #best_learning_rate_value_fan = np.array([1e-6])
     
    #best_batch_size_value_slider = np.array([32])
    #best_learning_rate_value_slider = np.array([1e-6])
    
    sampling_rate_sound = 16000
    
    kf = KFold(n_splits=5)
    dataset_splits = list(enumerate(kf.split(X_train, y_train)))
    
    for learning_rate in best_learning_rate_value_pump:
            for batch_size in best_batch_size_value_pump:
                
                result = []  
                print('---' * 20)
                print(f'Running for learning_rate={learning_rate}, batch_size={batch_size}')
                print(f'KFold\t|AUC\t|')

                # Validação cruzada K-Fold
                for i, (train_index, val_index) in dataset_splits:
                    # Registro do tempo de início do experimento
                    start = time.time() / 60.0

                    # Instancia o modelo GANF com os parâmetros atuais
                    model_GANF = GANF(
                        sampling_rate=sampling_rate_sound,
                        mono=True,
                        use_array2mfcc=True,
                        isForWaveData=True,
                    )

                    # Seleção das amostras de treino para o fold atual
                    X_train_KFold = X_train[train_index]
                    y_train_fold = y_train[train_index]

                    # Treinamento do modelo com os dados de treino do fold
                    model_GANF.fit(X_train_KFold, y_train_fold, batch_size=int(batch_size), learning_rate=learning_rate, epochs=20)

                    # Predição no conjunto de teste
                    auc = calculate_aucroc(model_GANF, X_test, y_test)

                    # Registro do tempo de término do experimento
                    end = time.time() / 60.0

                    # Cálculo da duração da execução
                    execution_duration = end - start
                    
                    # Armazena os resultados de AUC e os parâmetros do modelo para cada fold
                    result.append([model_name, i, batch_size, learning_rate, execution_duration, auc])  
                    print(f'K({i}):\t|{auc:0.5f}\t|')

                    # Salvar resultados em CSV para o fold atual
                    fold_csv_filename = os.path.join(save_dir, f'{machine_type}_fold-{i}_batch_size-{batch_size}_learning_rate-{learning_rate}.csv')

                    # Cria DataFrame com os resultados e salva
                    df_fold_results = pd.DataFrame(result, columns=['Model', 'Fold', 'batch_size', 'learning_rate', 'Execution_Duration', 'AUC_ROC'])
                    df_fold_results.to_csv(fold_csv_filename, sep=',', encoding='utf-8', index=False)

                    print(f'Fold {i} results saved to {fold_csv_filename}')
    
    
def info(title):
    print(title)
    print("module name:", __name__)
    print("parent process:", os.getppid())
    print("process id:", os.getpid())

with torch.cuda.device(0): 
    if __name__ == "__main__":
        set_start_method("spawn")
        info("main line")
        p = Process(target=run_ganf_experiment)
        p.start()
        p.join()