import os
import tensorflow as tf
import pandas as pd
import numpy as np 
import sys
import time
from multiprocessing import Process

import matplotlib.pyplot as plt
import scipy.stats as st
import librosa
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from mtsa.metrics import calculate_aucroc
from mtsa import files_train_test_split
from mtsa import IForest
from mtsa import Hitachi

module_path = os.path.abspath(os.path.join('../mtsa/'))
if module_path not in sys.path:
    sys.path.append(module_path)

def run_iforest_experiment():
    """
    Executa experimentos de detecção de anomalias de séries temporais multivariadas acústicas usando Isolation Forest.
    
    Esta função configura os dispositivos GPU, define os caminhos dos dados para diferentes objetos
    (fan, pump, slider, valve), configura diferentes conjuntos de parâmetros para Isolation Forest,
    divide os dados em conjuntos de treino e teste, e realiza validação cruzada K-Fold para avaliar
    o desempenho do modelo. Os resultados dos experimentos são salvos em arquivos CSV para análise posterior.
    """

    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Definição dos caminhos de entrada para diferentes objetos e IDs no conjunto de dados MIMII
    path_input_fan_id_00 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "fan", "id_00")
    path_input_fan_id_02 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "fan", "id_02")
    path_input_fan_id_04 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "fan", "id_04")
    path_input_fan_id_06 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "fan", "id_06")

    path_input_pump_id_00 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "pump", "id_00")
    path_input_pump_id_02 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "pump", "id_02")
    path_input_pump_id_04 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "pump", "id_04")
    path_input_pump_id_06 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "pump", "id_06")

    path_input_slider_id_00 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "slider", "id_00")
    path_input_slider_id_02 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "slider", "id_02")
    path_input_slider_id_04 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "slider", "id_04")
    path_input_slider_id_06 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "slider", "id_06")

    path_input_valve_id_00 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "valve", "id_00")
    path_input_valve_id_02 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "valve", "id_02")
    path_input_valve_id_04 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "valve", "id_04")
    path_input_valve_id_06 = os.path.join(os.getcwd(),  "..", "..", "MIMII", "valve", "id_06")

    # Define os caminhos de dados a serem utilizados nos experimentos
    datapaths = np.array([ 
        [path_input_fan_id_06,'path_input_fan_id_06']
        ])  
    # Definição de diferentes experimentos com variações nos parâmetros do Isolation Forest  
    # Cada experimento é um array contendo: [n_estimators, contamination, max_samples, max_features]
    experiment01 = np.array([100, 0.1, 128, 0.5])
    experiment02 = np.array([100, 0.1, 128, 0.8])
    experiment03 = np.array([100, 0.1, 128, 1.0])

    experiment04 = np.array([100, 0.1, 256, 0.5])
    experiment05 = np.array([100, 0.1, 256, 0.8])
    experiment06 = np.array([100, 0.1, 256, 1.0])

    experiment07 = np.array([5, 0.1, 128, 0.5])
    experiment08 = np.array([5, 0.1, 128, 0.8])
    experiment09 = np.array([5, 0.1, 128, 1.0])

    experiment10 = np.array([5, 0.1, 256, 0.5])
    experiment11 = np.array([5, 0.1, 256, 0.8])
    experiment12 = np.array([5, 0.1, 256, 1.0])

    experiment13 = np.array([10, 0.1, 128, 0.5])
    experiment14 = np.array([10, 0.1, 128, 0.8])
    experiment15 = np.array([10, 0.1, 128, 1.0])

    experiment16 = np.array([10, 0.1, 256, 0.5])
    experiment17 = np.array([10, 0.1, 256, 0.8])
    experiment18 = np.array([10, 0.1, 256, 1.0])

    experiment19 = np.array([30, 0.1, 128, 0.5])
    experiment20 = np.array([30, 0.1, 128, 0.8])
    experiment21 = np.array([30, 0.1, 128, 1.0])

    experiment22 = np.array([30, 0.1, 256, 0.5])
    experiment23 = np.array([30, 0.1, 256, 0.8])
    experiment24 = np.array([30, 0.1, 256, 1.0])

    experiment25 = np.array([50, 0.1, 128, 0.5])
    experiment26 = np.array([50, 0.1, 128, 0.8])
    experiment27 = np.array([50, 0.1, 128, 1.0])

    experiment28 = np.array([50, 0.1, 256, 0.5])
    experiment29 = np.array([50, 0.1, 256, 0.8])
    experiment30 = np.array([50, 0.1, 256, 1.0])   

    experiment31 = np.array([1, 0.1, 128, 0.5])
    experiment32 = np.array([1, 0.1, 128, 0.8])

    experiment33 = np.array([3, 0.1, 128, 0.5])
    experiment34 = np.array([3, 0.1, 128, 0.8])

    experiment35 = np.array([5, 0.1, 128, 0.5])
    experiment36 = np.array([5, 0.1, 128, 0.8])

    experiment37 = np.array([7, 0.1, 128, 0.5])
    experiment38 = np.array([7, 0.1, 128, 0.8])

    experiment39 = np.array([9, 0.1, 128, 0.5]) 
    experiment40 = np.array([9, 0.1, 128, 0.8]) 

    # Lista dos experimentos a serem executados
    experiments = np.array([        
         experiment33,         
         experiment34,        
         ])    
    
    # Divisão dos dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = files_train_test_split(path_input_fan_id_06)

    # Conversão dos dados de treino para array NumPy
    X = np.array(X_train)

    # Configuração da validação cruzada K-Fold com 10 partições
    k=10
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    kf.get_n_splits(X)

    # Loop sobre cada caminho de dados 
    for data in datapaths:
        # Loop sobre cada conjunto de parâmetros de experimento
        for parameter in experiments:
            
            result=[] # Lista para armazenar os resultados de cada fold
            print('---'*20)
            print('KFold\t|AUC\t|')
            for i, (train_index, val_index) in enumerate(kf.split(X)):
                
                # Registro do tempo de início do experimento (em minutos)
                start = time.time() / 60.0

                # Instancia o modelo IForest com os parâmetros atuais
                model_iforest = IForest(n_estimators=int(parameter[0]), contamination=parameter[1], max_samples=int(parameter[2]), max_features=parameter[3])

                # Seleção das amostras de treino para o fold atual
                X_train_KFold = X[train_index]
                
                
                
                # Treinamento do modelo com os dados de treino do fold
                model_iforest.fit(X_train_KFold)

                # Predição no conjunto de validação
                #preditions = model_iforest.predict(X_test)

                # Avaliação dos resultados
                auc = calculate_aucroc(model_iforest, X_test, y_test)

                # Registro do tempo de término do experimento (em minutos)
                end = time.time() / 60.0

                # Cálculo da duração da execução
                execution_duration = end - start

                # Criação do DataFrame de experimentos com os resultados atuais
                experiment_dataframe = model_iforest.get_experiment_dataframe(data[1], f'n_estimator-{parameter[0]}_contamination-{parameter[1]}_max_samples-{parameter[2]}_max_features-{parameter[3]}', execution_duration)
                experiment_dataframe['AUC_ROC'] = auc
                experiment_dataframe['execution_duration'] = execution_duration

                # Verificação se o arquivo CSV para este conjunto de parâmetros já existe
                csv_filename = f'n_estimator-{parameter[0]}_contamination-{parameter[1]}_max_samples-{parameter[2]}_max_features-{parameter[3]}.csv'
                if os.path.isfile(csv_filename):
                    # Se existir, lê o CSV existente, coloca os novos resultados e salva novamente
                    df_existente = pd.read_csv(f'n_estimator-{parameter[0]}_contamination-{parameter[1]}_max_samples-{parameter[2]}_max_features-{parameter[3]}.csv')
                    experiment_dataframe = pd.concat([df_existente, experiment_dataframe], ignore_index=True)
                    experiment_dataframe.to_csv(f'n_estimator-{parameter[0]}_contamination-{parameter[1]}_max_samples-{parameter[2]}_max_features-{parameter[3]}.csv', sep=',', encoding='utf-8')
                else:
                    experiment_dataframe.to_csv(f'n_estimator-{parameter[0]}_contamination-{parameter[1]}_max_samples-{parameter[2]}_max_features-{parameter[3]}.csv', sep=',', encoding='utf-8')
                
                result.append([auc])  
                print( f'K({i}):\t|{auc:0.5f}\t|' )
                print('---'*20)

'''
    for data in datapaths:
        
        result=[]
        print('---'*20)
        print('KFold\t|RMSE\t|Variance score\t|AUC\t|')
        for i, (train_index, val_index) in enumerate(kf.split(X)):
        
            start = time.time()

            model_hitachi = Hitachi()

            X_train_KFold = X[train_index]
            
            
            # Treinamento do modelo
            model_hitachi.fit(X_train_KFold)

            # Predição no conjunto de validação
            preditions = model_hitachi.predict(X_test)

            # Avaliação dos resultados
            ### COMENZAR O CODIGO AQUI ### 
            rmse = mean_squared_error(y_test, preditions, squared=False)
            score = r2_score(y_test, preditions)
            auc = calculate_aucroc(model_hitachi, X_test, y_test)

            end = time.time()

            execution_duration = end - start

            experiment_dataframe = model_iforest.get_experiment_dataframe(data[1], f'Hitachi', execution_duration)
            experiment_dataframe['AUC_ROC'] = auc
            experiment_dataframe['RMSE'] = rmse
            experiment_dataframe['Score'] = score
            experiment_dataframe['execution_duration'] = execution_duration

            if os.path.isfile(f'hita.csv'):
                df_existente = pd.read_csv(f'Hitachi.csv')
                experiment_dataframe = pd.concat([df_existente, experiment_dataframe], ignore_index=True)
                experiment_dataframe.to_csv(f'Hitachi.csv', sep=',', encoding='utf-8')
            else:
                experiment_dataframe.to_csv(f'Hitachi.csv', sep=',', encoding='utf-8')
            
            result.append([rmse,score,auc])  
            print( f'K({i}):\t|{rmse:0.5f}\t|{score:0.5f}\t|{auc:0.5f}\t|' )
            print('---'*20)
            '''
            
def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

if __name__ == '__main__':
    info('main line')
    p = Process(target=run_iforest_experiment)
    p.start()
    p.join()