import os
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import KFold
from mtsa import IForest, files_train_test_split
from mtsa.metrics import calculate_aucroc

class ExperimentRunner:
    """
    Classe para rodar experimentos usando Isolation Forest. Permite rodar experimentos em dados de IDs individuais ou dados concatenados
    de todos os IDs de uma máquina.
    """

    def __init__(self):
        # Definição dos caminhos de entrada para diferentes IDs de cada máquina (fan, pump, slider, valve)
        self.all_machine_paths = {
            "fan": [
                os.path.join(os.getcwd(), "..", "..", "MIMII", "fan", "id_00"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "fan", "id_02"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "fan", "id_04"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "fan", "id_06")
            ],
            "pump": [
                os.path.join(os.getcwd(), "..", "..", "MIMII", "pump", "id_00"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "pump", "id_02"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "pump", "id_04"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "pump", "id_06")
            ],
            "slider": [
                os.path.join(os.getcwd(), "..", "..", "MIMII", "slider", "id_00"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "slider", "id_02"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "slider", "id_04"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "slider", "id_06")
            ],
            "valve": [
                os.path.join(os.getcwd(), "..", "..", "MIMII", "valve", "id_00"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "valve", "id_02"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "valve", "id_04"),
                os.path.join(os.getcwd(), "..", "..", "MIMII", "valve", "id_06")
            ]
        }

    def load_and_concatenate_data(self, paths):
        """
        Carrega os dados de diferentes IDs e os concatena em um único array.
        
        Argumentos:
        - paths (list): Lista de caminhos dos dados de diferentes IDs de uma máquina.

        Retorna:
        - X_train_concat, X_test_concat, y_train_concat, y_test_concat: Arrays concatenados de todos os IDs.
        """
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []
        
        for path in paths:
            X_train, X_test, y_train, y_test = files_train_test_split(path)
            X_train_list.append(X_train)
            X_test_list.append(X_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)
        
        X_train_concat = np.concatenate(X_train_list, axis=0)
        X_test_concat = np.concatenate(X_test_list, axis=0)
        y_train_concat = np.concatenate(y_train_list, axis=0)
        y_test_concat = np.concatenate(y_test_list, axis=0)
        
        return X_train_concat, X_test_concat, y_train_concat, y_test_concat

    def run_experiment_per_id(self, machine, id):
        """
        Executa experimentos em cada ID da máquina, usando K-Fold para cada ID individualmente.
        
        Argumentos:
        - machine (str): Nome da máquina (fan, pump, slider, valve).
        - id (str): ID específico a ser rodado (id_00, id_02, id_04, id_06).
        """
        # Verifica se a máquina é válida
        if machine not in self.all_machine_paths:
            raise ValueError(f"Máquina {machine} não encontrada. Escolha entre: {list(self.all_machine_paths.keys())}")
        
        # Filtra o caminho do ID específico
        selected_path = None
        for path in self.all_machine_paths[machine]:
            if id in path:
                selected_path = path
                break
        
        if selected_path is None:
            raise ValueError(f"ID {id} não encontrado para a máquina {machine}.")
        
        # Carrega os dados para o ID específico
        X_train, X_test, y_train, y_test = files_train_test_split(selected_path)
        print(f"Running K-Fold on machine: {machine}, ID: {id}")
        
        # Executa o K-Fold para o ID específico
        self.run_kfold_experiment(X_train, X_test, y_train, y_test, machine, id)

    def run_experiment_on_concatenated_data(self, machine):
        """
        Executa experimentos com os dados concatenados de todos os IDs de uma máquina e aplicando K-Fold.
        """
        paths = self.all_machine_paths[machine]
        X_train, X_test, y_train, y_test = self.load_and_concatenate_data(paths)
        print(f"Running K-Fold on concatenated data for machine: {machine}")
        self.run_kfold_experiment(X_train, X_test, y_train, y_test, machine, "all_ids")

    def run_kfold_experiment(self, X_train, X_test, y_train, y_test, machine, dataset_label):
        """
        Executa a validação cruzada K-Fold e treinamento do modelo Isolation Forest nos dados fornecidos.
        """
        experiments = np.array([        
            [50, 0.1, 256, 1.0], [100, 0.1, 256, 1.0],
            [200, 0.1, 256, 1.0], [100, 0.1, 128, 1.0],
            [50, 0.1, 128, 1.0], [200, 0.1, 128, 1.0]
        ])
        X = np.array(X_train)
        k = 5
        kf = KFold(n_splits=k, shuffle=True, random_state=1)

        for parameter in experiments:
            result = []
            print(f'Running K-Fold with parameters: {parameter}')
            for i, (train_index, val_index) in enumerate(kf.split(X)):
                start = time.time() / 60.0
                model_iforest = IForest(n_estimators=int(parameter[0]), contamination=parameter[1],
                                        max_samples=int(parameter[2]), max_features=parameter[3])
                # Divide os dados de treino no fold atual
                X_train_KFold = X[train_index]
                y_train_KFold = y_train[train_index]
                
                # Treina o modelo no fold atual
                model_iforest.fit(X_train_KFold)
                
                # Testa o modelo no conjunto de teste completo
                auc = calculate_aucroc(model_iforest, X_test, y_test)
                
                end = time.time() / 60.0
                execution_duration = end - start

                # Criação do DataFrame de resultados
                experiment_dataframe = model_iforest.get_experiment_dataframe(
                    f'{machine}_{dataset_label}', 
                    f'n_estimator-{parameter[0]}_contamination-{parameter[1]}_max_samples-{parameter[2]}_max_features-{parameter[3]}',
                    execution_duration)
                experiment_dataframe['AUC_ROC'] = auc
                experiment_dataframe['execution_duration'] = execution_duration
                
                # Salvar os resultados no arquivo CSV
                csv_filename = f'{machine}_{dataset_label}_n_estimator-{parameter[0]}_contamination-{parameter[1]}.csv'
                if os.path.isfile(csv_filename):
                    df_existente = pd.read_csv(csv_filename)
                    experiment_dataframe = pd.concat([df_existente, experiment_dataframe], ignore_index=True)
                experiment_dataframe.to_csv(csv_filename, sep=',', encoding='utf-8')

if __name__ == '__main__':
    experiment_runner = ExperimentRunner()

    # Escolha a máquina e o ID
    machine_list = ["fan", "pump", "slider", "valve"]
    id = "id_00"  

    # Rodar o experimento para cada máquina com o ID selecionado
    for machine in machine_list:
        experiment_runner.run_experiment_per_id(machine, id)
