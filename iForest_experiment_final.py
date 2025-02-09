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
    
    # Valores para n_estimators e max_features
    n_estimators_values = [10]
    max_features_values = [0.5]
    contamination = 0.01
    max_samples = 128
    
    # Configuração da validação cruzada K-Fold
    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    
    # Pasta para salvar os resultados
    save_dir = os.path.join(os.getcwd(), 'experiment_final')
    os.makedirs(save_dir, exist_ok=True)

    # Loop sobre os IDs da máquina (com base na escolha feita pelo usuário)
    for machine_id in machine_ids:
        # Gerar o caminho dos dados para o ID atual
        data_path = get_data_path(machine_type, machine_id)
        print(f"Running experiments for {machine_type} - {machine_id}")
        
        # Divisão dos dados em conjuntos de treino e teste
        X_train, X_test, y_train, y_test = files_train_test_split(data_path)
        X = np.array(X_train)

        # Loop sobre max_features e n_estimators (Eixo Y = max_features e Eixo X = n_estimators)
        for max_features in max_features_values:
            for n_estimators in n_estimators_values:
                
                result = []  # Lista para armazenar os resultados de cada fold
                print('---' * 20)
                print(f'Running for max_features={max_features}, n_estimators={n_estimators}')
                print(f'KFold\t|AUC\t|')

                # Validação cruzada K-Fold
                for i, (train_index, val_index) in enumerate(kf.split(X)):
                    # Registro do tempo de início do experimento (em minutos)
                    start = time.time() / 60.0

                    # Instancia o modelo IForest com os parâmetros atuais
                    model_iforest = IForest(n_estimators=n_estimators, contamination=contamination,
                                            max_samples=max_samples, max_features=max_features)

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
                    result.append([i, n_estimators, contamination, max_samples, max_features, execution_duration, auc])  
                    print(f'K({i}):\t|{auc:0.5f}\t|')

                # Salvar resultados em CSV
                experiment_name = f'{machine_type}_{machine_id}_n_estimators-{n_estimators}_max_features-{max_features}_contamination-{contamination}_max_samples-{max_samples}'
                csv_filename = os.path.join(save_dir, f'{experiment_name}.csv')

                # Cria DataFrame com os resultados e salva
                df_results = pd.DataFrame(result, columns=['Fold', 'n_estimators', 'contamination', 'max_samples', 'max_features', 'Execution_Duration', 'AUC_ROC'])
                df_results.to_csv(csv_filename, sep=',', encoding='utf-8', index=False)

                print(f'Results saved to {csv_filename}')
                print('---' * 20)
            
if __name__ == '__main__':
    run_iforest_experiment(machine_type="slider", test_all_ids=False, specific_id="id_00")
