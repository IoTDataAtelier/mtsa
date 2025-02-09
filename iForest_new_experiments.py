import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from mtsa import IForest, files_train_test_split
from mtsa.metrics import calculate_aucroc

def run_iforest_experiment(machine_type=None, test_all_ids=True, specific_id=None):
    """
    Executa experimentos de detecção de anomalias de séries temporais multivariadas acústicas usando Isolation Forest.
    Parâmetros:
        - machine_type (str): O tipo de máquina a ser testada (fan, pump, slider, valve).
        - test_all_ids (bool): Se True, testa todos os IDs da máquina. Se False, usa apenas um ID específico.
        - specific_id (str): ID específico a ser testado. Relevante apenas quando test_all_ids=False.
    """

    # Definição dos tipos de máquinas e seus respectivos IDs
    machines = {
        "fan": ["id_00", "id_02", "id_04", "id_06"],
        "pump": ["id_00", "id_02", "id_04", "id_06"],
        "slider": ["id_00", "id_02", "id_04", "id_06"],
        "valve": ["id_00", "id_02", "id_04", "id_06"]
    }

    # Verifica se o tipo de máquina é válido
    if machine_type not in machines:
        raise ValueError(f"Máquina {machine_type} não encontrada. Escolha entre: {list(machines.keys())}")
    
    # Obtém os IDs da máquina
    machine_ids = machines[machine_type]

    # Se não for para testar todos os IDs, verifica se um ID específico foi passado
    if not test_all_ids and specific_id:
        if specific_id in machine_ids:
            machine_ids = [specific_id]
        else:
            raise ValueError(f"O ID {specific_id} não é válido para a máquina {machine_type}. IDs válidos: {machine_ids}")
    
    # Função para gerar o caminho de dados com base no tipo de máquina e ID
    def get_data_path(machine_type, machine_id):
        return os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, machine_id)
    
    # Definição de diferentes experimentos com variações nos parâmetros do Isolation Forest  
    # Cada experimento é um array contendo: [n_estimators, contamination, max_samples, max_features]
    experiment01 = np.array([50, 0.01, 128, 1.0])
    experiment02 = np.array([50, 0.01, 256, 1.0])
    
    experiment03 = np.array([100, 0.01, 256, 1.0])
    experiment04 = np.array([100, 0.01, 512, 1.0])
    experiment05 = np.array([100, 0.05, 256, 1.0])
    experiment06 = np.array([100, 0.05, 512, 1.0])
    experiment07 = np.array([100, 0.05, 512, 0.75])
    experiment08 = np.array([100, 0.05, 512, 0.5])

    experiment09 = np.array([200, 0.01, 128, 1.0])
    experiment10 = np.array([200, 0.01, 256, 1.0])
    experiment11 = np.array([200, 0.01, 512, 1.0])

    experiment12 = np.array([500, 0.01, 128, 1.0])
    experiment13 = np.array([500, 0.01, 256, 1.0])
    experiment14 = np.array([500, 0.01, 512, 1.0])
    
    # Após primeira análise, melhorando os experimentos
    experiment15 = np.array([10, 0.01, 512, 1.0])
    experiment27 = np.array([40, 0.01, 512, 1.0])
    experiment18 = np.array([70, 0.01, 512, 1.0])
    experiment16 = np.array([100, 0.01, 512, 1.0])
    experiment17 = np.array([150, 0.01, 512, 1.0])
    
    experiment19 = np.array([10, 0.01, 128, 1.0])
    experiment28 = np.array([40, 0.01, 128, 1.0])
    experiment22 = np.array([70, 0.01, 128, 1.0])
    experiment20 = np.array([100, 0.01, 128, 1.0])
    experiment21 = np.array([150, 0.01, 128, 1.0])
    
    experiment23 = np.array([10, 0.01, 512, 0.5])
    experiment29 = np.array([40, 0.01, 512, 0.5])
    experiment26 = np.array([70, 0.01, 512, 0.5])
    experiment24 = np.array([100, 0.01, 512, 0.5])
    experiment25 = np.array([150, 0.01, 512, 0.5])
    
    experiment30 = np.array([10, 0.14, 512, 1.0])
    experiment31 = np.array([40, 0.14, 512, 1.0])
    experiment34 = np.array([70, 0.14, 512, 1.0])
    experiment32 = np.array([100, 0.14, 512, 1.0])
    experiment33 = np.array([150, 0.14, 512, 1.0])
    
    #Fan 
    experiment35 = np.array([10, 0.14, 512, 1.0])
    experiment36 = np.array([40, 0.14, 512, 1.0])
    experiment37 = np.array([70, 0.14, 512, 1.0])
    experiment38 = np.array([100, 0.14, 512, 1.0])
    experiment39 = np.array([150, 0.14, 512, 1.0])
    
    #Pump 
    experiment40 = np.array([10, 0.01, 128, 0.5])
    experiment41 = np.array([10, 0.12, 128, 0.5])
    
    experiment42 = np.array([40, 0.01, 128, 0.75])
    experiment43 = np.array([40, 0.01, 64, 1.0])
    experiment44 = np.array([40, 0.12, 64, 1.0])
    experiment45 = np.array([40, 0.12, 128, 1.0])
    
    experiment46 = np.array([70, 0.01, 128, 0.5])
    experiment47 = np.array([70, 0.01, 64, 1.0])
    experiment48 = np.array([70, 0.12, 64, 1.0])
    experiment49 = np.array([70, 0.12, 128, 0.5])
    experiment50 = np.array([70, 0.12, 512, 0.5])
    
    experiment51 = np.array([100, 0.01, 64, 0.5])
    experiment52 = np.array([100, 0.12, 128, 1.0])
    
    experiment53 = np.array([10, 0.01, 32, 1.0])
    experiment55 = np.array([2, 0.01, 128, 1.0])
    experiment56 = np.array([2, 0.01, 32, 1.0])
    experiment57 = np.array([2, 0.01, 64, 0.5])
    experiment58 = np.array([10, 0.01, 32, 0.5])
    
    experiment_best_hyperparameter_fan = np.array([100, 0.01, 256, 1.0])
    experiment_best_hyperparameter_pump = np.array([70, 0.01, 128, 1.0])
    experiment_best_hyperparameter_pumpTest = np.array([70, 0.01, 1006, 1.0])
    experiment_best_hyperparameter_pumpTest2 = np.array([70, 0.01, 32, 1.0])
    experiment_best_hyperparameter_valve = np.array([70, 0.01, 256, 0.5])
    experiment_best_hyperparameter_valve2 = np.array([100, 0.01, 512, 1.0])
    experiment_best_hyperparameter_slider = np.array([10, 0.5, 128, 1.0])

    # Definição de diferentes experimentos com variações nos parâmetros do Isolation Forest  
    experiments = np.array([ 
        experiment_best_hyperparameter_pumpTest2          
    ])

    # Configuração da validação cruzada K-Fold
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    
    # Pasta para salvar os resultados
    save_dir = os.path.join(os.getcwd(), 'experiments_hyperparameter')
    os.makedirs(save_dir, exist_ok=True)

    # Loop sobre os IDs da máquina (com base na escolha feita pelo usuário)
    for machine_id in machine_ids:
        # Gerar o caminho dos dados para o ID atual
        data_path = get_data_path(machine_type, machine_id)
        print(f"Running experiments for {machine_type} - {machine_id}")
        
        # Divisão dos dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = files_train_test_split(data_path)
        X = np.array(X_train)

        # Loop sobre cada conjunto de parâmetros de experimento
        for parameter in experiments:
            
            result = []  # Lista para armazenar os resultados de cada fold
            print('---' * 20)
            print(f'KFold\t|AUC\t|')

            # Validação cruzada K-Fold
            for i, (train_index, val_index) in enumerate(kf.split(X)):
                # Registro do tempo de início do experimento (em minutos)
                start = time.time() / 60.0

                # Instancia o modelo IForest com os parâmetros atuais
                model_iforest = IForest(n_estimators=int(parameter[0]), contamination=parameter[1],
                                        max_samples=int(parameter[2]), max_features=parameter[3])

                # Seleção das amostras de treino para o fold atual
                X_train_KFold = X[train_index]

                # Treinamento do modelo com os dados de treino do fold
                model_iforest.fit(X_train_KFold)

                # Predição no conjunto de teste
                auc = calculate_aucroc(model_iforest, X_test, y_test)

                # Registro do tempo de término do experimento (em minutos)
                end = time.time() / 60.0

                # Cálculo da duração da execução
                execution_duration = end - start
                
                # Armazena os resultados de AUC e os parâmetros do modelo para cada fold
                result.append([i, int(parameter[0]), parameter[1], int(parameter[2]), parameter[3], execution_duration, auc])  
                print(f'K({i}):\t|{auc:0.5f}\t|')

            # Salvar resultados em CSV
            experiment_name = f'{machine_type}_{machine_id}_n_estimators-{parameter[0]}_contamination-{parameter[1]}_max_samples-{parameter[2]}_max_features-{parameter[3]}'
            csv_filename = os.path.join(save_dir, f'{experiment_name}.csv')

            # Cria DataFrame com os resultados e salva
            df_results = pd.DataFrame(result, columns=['Fold', 'n_estimators', 'contamination', 'max_samples', 'max_features','Execution_Duration', 'AUC_ROC'])
            df_results.to_csv(csv_filename, sep=',', encoding='utf-8', index=False)

            print(f'Results saved to {csv_filename}')
            print('---' * 20)
            
if __name__ == '__main__':
    
    #run_iforest_experiment(machine_type="fan", test_all_ids=True)
    run_iforest_experiment(machine_type="pump", test_all_ids=False, specific_id="id_00")
