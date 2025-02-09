# model_ood_experiment.py

import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, OutlierMixin
from mtsa import IForest, files_train_test_split
from mtsa.metrics import calculate_aucroc
from itertools import product
from copy import deepcopy
from mtsa import Hitachi

class ModelOODExperiment:
    """
    Classe para executar experimentos "out of distribution" para múltiplos modelos com diferentes hiperparâmetros.

    Cada modelo é treinado com dados normais de três IDs e testado no ID restante.
    """

    def __init__(self, machine_type, train_ids, test_id, models, k=5):
        """
        Inicializa o experimento.

        Parâmetros:
            - machine_type (str): O tipo de máquina (fan, pump, slider, valve).
            - train_ids (list): IDs de treinamento (Ex: ["id_02", "id_04", "id_06"]).
            - test_id (str): ID de teste (Ex: "id_00").
            - models (list): Lista de dicionários contendo modelos e seus hiperparâmetros.
                             Exemplo:
                             [
                                 {
                                     "name": "Hitachi",
                                     "instance": Hitachi(),
                                     "hyperparameters": {
                                         "batch_size": [1024, 512, 256, 128, 64, 32],
                                         "learning_rate": [1e-3, 1e-6]
                                     }
                                 },
                                 ...
                             ]
            - k (int): Número de folds para validação cruzada.
        """
        self.machine_type = machine_type
        self.train_ids = train_ids
        self.test_id = test_id
        self.models = models
        self.k = k
        self.save_dir = os.path.join(os.getcwd(), 'experiment_ood', self.machine_type.upper())
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data(self, machine_type, ids):
        """
        Carrega os dados para os IDs fornecidos.

        Parâmetros:
            - machine_type (str): O tipo de máquina.
            - ids (list): Lista de IDs cujos dados serão carregados.
        """
        X, y = [], []
        for machine_id in ids:
            data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, machine_id)
            if not os.path.exists(data_path):
                print(f"Warning: Data path '{data_path}' does not exist. Skipping.")
                continue
            X_data, _, y_data, _ = files_train_test_split(data_path)  
            X.extend(X_data)
            y.extend(y_data)
        return np.array(X), np.array(y)
    
    def load_test_data(self, machine_type, test_id):
        """
        Carrega os dados de teste para o ID fornecido.

        Parâmetros:
            - machine_type (str): O tipo de máquina.
            - test_id (str): ID de teste para o qual os dados serão carregados.

        Retorna:
            - X_test (np.ndarray): Dados de teste.
            - y_test (np.ndarray): Rótulos de teste correspondentes.
        """
        data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, test_id)
        
        if not os.path.exists(data_path):
            print(f"Error: Test data path '{data_path}' does not exist.")
            return np.array([]), np.array([])
        
        _, X_test, _, y_test = files_train_test_split(data_path)
        
        return X_test, y_test
    
    def run(self):
        """
        Executa o experimento OOD para todos os modelos e combinações de hiperparâmetros.
        """
        print(f"Running OOD experiment for {self.machine_type} - Train IDs: {self.train_ids}, Test ID: {self.test_id}")

        # Carregar dados de treino (apenas normais)
        X_train, _ = self.load_data(self.machine_type, self.train_ids)
        print(f"Loaded {len(X_train)} training samples for machine_type='{self.machine_type}' with train_ids={self.train_ids}")

        # Carregar dados de teste
        X_test, y_test = self.load_test_data(self.machine_type, self.test_id)
        print(f"Loaded {len(X_test)} test samples for machine_type='{self.machine_type}' with test_id='{self.test_id}'")

        # Verificar se os dados foram carregados corretamente
        if len(X_train) == 0:
            print("Error: No training data loaded. Exiting experiment.")
            return
        if len(X_test) == 0:
            print("Error: No test data loaded. Exiting experiment.")
            return

        # Configuração da validação cruzada K-Fold
        kf = KFold(n_splits=self.k, shuffle=True, random_state=1)

        # Iterar sobre cada modelo
        for model_dict in self.models:
            model_name = model_dict["name"]
            model_instance = model_dict["instance"]
            hyperparams = model_dict["hyperparameters"]

            # Gerar todas as combinações de hiperparâmetros
            hyperparam_names = list(hyperparams.keys())
            hyperparam_values = list(hyperparams.values())
            hyperparam_combinations = list(product(*hyperparam_values))

            print(f"\n=== Starting experiments for model: {model_name} ===")

            # Iterar sobre cada combinação de hiperparâmetros
            for hyperparam_combo in hyperparam_combinations:
                hyperparam_dict = dict(zip(hyperparam_names, hyperparam_combo))
                batch_size = hyperparam_dict.get("batch_size", 512)
                learning_rate = hyperparam_dict.get("learning_rate", 1e-3)

                result = []  # Lista para armazenar os resultados de cada fold
                print('---' * 20)
                print(f'Running for batch_size={batch_size}, learning_rate={learning_rate}')
                print(f'Fold\t|AUC_ROC\t|Execution_Time(min)\t|')

                # Validação cruzada K-Fold
                for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
                    # Registro do tempo de início do experimento
                    start = time.time()

                    # Instanciação do modelo com os hiperparâmetros atuais
                    # Assumindo que o modelo pode receber batch_size e learning_rate como argumentos
                    # Se não, você pode precisar ajustar a instância do modelo aqui
                    model = self._clone_model(model_instance, hyperparam_dict)

                    # Seleção das amostras de treino para o fold atual
                    X_train_fold = X_train[train_index]

                    # Treinamento do modelo com os dados de treino do fold
                    model.fit(X_train_fold)

                    # Predição no conjunto de teste
                    auc = calculate_aucroc(model, X_test, y_test)

                    # Registro do tempo de término do experimento
                    end = time.time()

                    # Cálculo da duração da execução em minutos
                    execution_duration = (end - start) / 60.0
                    
                    # Armazena os resultados de AUC e os parâmetros do modelo para cada fold
                    result.append([fold + 1, batch_size, learning_rate, execution_duration, auc])  
                    print(f'Fold {fold + 1}\t|{auc:0.5f}\t|{execution_duration:0.2f}\t|')

                # Salvar resultados em CSV
                experiment_name = f'{self.machine_type}_OOD_{model_name}_train-{self.train_ids}_test-{self.test_id}_batch_size-{batch_size}_learning_rate-{learning_rate}'
                csv_filename = os.path.join(self.save_dir, f'{experiment_name}.csv')

                # Cria DataFrame com os resultados e salva
                df_results = pd.DataFrame(result, columns=['Fold', 'batch_size', 'learning_rate', 'Execution_Duration(min)', 'AUC_ROC'])
                df_results.to_csv(csv_filename, sep=',', encoding='utf-8', index=False)

                print(f'Results saved to {csv_filename}')
                print('---' * 20)
    
    def _clone_model(self, model_instance, hyperparam_dict):
        """
        Clona o modelo com os hiperparâmetros atualizados.

        Parâmetros:
            - model_instance (BaseEstimator): Instância do modelo.
            - hyperparam_dict (dict): Dicionário de hiperparâmetros.

        Retorna:
            - cloned_model (BaseEstimator): Novo modelo com os hiperparâmetros atualizados.
        """
        cloned_model = deepcopy(model_instance)
        for param, value in hyperparam_dict.items():
            setattr(cloned_model, param, value)
        # Reconstruir o pipeline/modelo com os novos hiperparâmetros
        cloned_model.model = cloned_model._build_model()
        return cloned_model

if __name__ == '__main__':

    machine_type = "fan"
    train_ids = ["id_02", "id_04", "id_06"]
    test_id = "id_00"
    batch_size_values = np.array([1024, 512, 256, 128, 64, 32])
    learning_rate_values = np.array([1e-3, 1e-6])

    # Definir os modelos e seus hiperparâmetros
    models = [
        {
            "name": "Hitachi",
            "instance": Hitachi(),
            "hyperparameters": {
                "batch_size": batch_size_values,
                "learning_rate": learning_rate_values
            }
        },
        # Adicione outros modelos aqui seguindo a mesma estrutura
        # Exemplo:
        # {
        #     "name": "GANF",
        #     "instance": GANF(),
        #     "hyperparameters": {
        #         "batch_size": batch_size_values,
        #         "learning_rate": learning_rate_values
        #     }
        # },
        # {
        #     "name": "RANSynCoders",
        #     "instance": RANSynCoders(),
        #     "hyperparameters": {
        #         "batch_size": batch_size_values,
        #         "learning_rate": learning_rate_values
        #     }
        # },
    ]

    ood_experiment = ModelOODExperiment(
        machine_type=machine_type,
        train_ids=train_ids,
        test_id=test_id,
        models=models,
        k=5
    )
    ood_experiment.run()
