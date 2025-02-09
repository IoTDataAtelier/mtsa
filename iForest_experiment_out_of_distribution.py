import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import KFold
from mtsa import IForest, files_train_test_split
from mtsa.metrics import calculate_aucroc

class IForestOODExperiment:
    """
    Classe para executar experimentos "out of distribution" utilizando Isolation Forest

    O modelo é treinado com dados normais de três IDs e testado no ID restante.
    """
    
    def __init__(self, machine_type, train_ids, test_id, max_features_values, n_estimators_values, k=5):
        """
        Inicializa o experimento.

        Parâmetros:
            - machine_type (str): O tipo de máquina (fan, pump, slider, valve).
            - train_ids (list): IDs de treinamento (Ex: ["id_02", "id_04", "id_06"]).
            - test_id (str): ID de teste (Ex: "id_00").
            - max_features_values (list): Lista de valores para max_features.
            - n_estimators_values (list): Lista de valores para n_estimators.
            - k (int): Número de folds para validação cruzada.
        """
        self.machine_type = machine_type
        self.train_ids = train_ids
        self.test_id = test_id
        self.max_features_values = max_features_values
        self.n_estimators_values = n_estimators_values
        self.k = k
        self.contamination = 0.01  
        self.max_samples = 256 
        self.model_name = 'Isolation_Forest' 
        self.save_dir = os.path.join(os.getcwd(), f'experiment_ood_{self.model_name}_bestHyperparameters', self.machine_type.upper())
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
            - X_test (list): Dados de teste (array de NumPy).
            - y_test (list): Rótulos de teste correspondentes (normal e anômalo).
        """
        data_path = os.path.join(os.getcwd(), "..", "..", "MIMII", machine_type, test_id)
        
        _, X_test, _, y_test = files_train_test_split(data_path)
        
        return X_test, y_test
    
    def run(self):
        """
        Executa o experimento de "out of distribution".
        """
        print(f"Running OOD experiment for {self.machine_type} - Train IDs: {self.train_ids}, Test ID: {self.test_id}")

        # Carregar dados de treino (apenas normais)
        X_train, _ = self.load_data(self.machine_type, self.train_ids)
        print(f"Loaded {len(X_train)} test samples for machine_type='{machine_type}'")

        # Carregar dados de teste
        X_test, y_test = self.load_test_data(self.machine_type, self.test_id)
        print(f"Loaded {len(X_test)} test samples for machine_type='{machine_type}'")

        # Configuração da validação cruzada K-Fold
        kf = KFold(n_splits=self.k, shuffle=True, random_state=1)

        # Loop sobre max_features e n_estimators
        for max_features in self.max_features_values:
            for n_estimators in self.n_estimators_values:
                
                result = []  # Lista para armazenar os resultados de cada fold
                print('---' * 20)
                print(f'Running for max_features={max_features}, n_estimators={n_estimators}')
                print(f'KFold\t|AUC\t|')

                # Validação cruzada K-Fold
                for i, (train_index, val_index) in enumerate(kf.split(X_train)):
                    # Registro do tempo de início do experimento
                    start = time.time() / 60.0

                    # Instancia o modelo IForest com os parâmetros atuais
                    model_iforest = IForest(n_estimators=n_estimators, contamination=self.contamination,
                                            max_samples=self.max_samples, max_features=max_features)

                    # Seleção das amostras de treino para o fold atual
                    X_train_KFold = X_train[train_index]

                    # Treinamento do modelo com os dados de treino do fold
                    model_iforest.fit(X_train_KFold)

                    # Predição no conjunto de teste
                    auc = calculate_aucroc(model_iforest, X_test, y_test)

                    # Registro do tempo de término do experimento
                    end = time.time() / 60.0

                    # Cálculo da duração da execução
                    execution_duration = end - start
                    
                    # Armazena os resultados de AUC e os parâmetros do modelo para cada fold
                    result.append([self.model_name, i, n_estimators, self.contamination, self.max_samples, max_features, execution_duration, auc])  
                    print(f'K({i}):\t|{auc:0.5f}\t|')

                # Salvar resultados em CSV
                experiment_name = f'{self.machine_type}_OOD_train-{self.train_ids}_test-{self.test_id}_n_estimators-{n_estimators}_max_features-{max_features}'
                csv_filename = os.path.join(self.save_dir, f'{experiment_name}.csv')

                # Cria DataFrame com os resultados e salva
                df_results = pd.DataFrame(result, columns=['Model', 'Fold', 'n_estimators', 'contamination', 'max_samples', 'max_features', 'Execution_Duration', 'AUC_ROC'])
                df_results.to_csv(csv_filename, sep=',', encoding='utf-8', index=False)

                print(f'Results saved to {csv_filename}')
                print('---' * 20)

if __name__ == '__main__':
    machine_type = "fan"
    train_ids = ["id_02"]
    test_id = "id_00"
    #max_features_values = [0.5, 1.0]
    #n_estimators_values = [2, 10, 40, 70, 100]
    
    #best_max_features_values_Valve = [1.0]
    #best_n_estimators_values_Valve = [100]
    
    #best_max_features_values_Pump = [1.0]
    #best_n_estimators_values_Pump = [70]
    
    best_max_features_values_Fan = [1.0]
    best_n_estimators_values_Fan = [100]
    
    #best_max_features_values_Slider = [1.0]
    #best_n_estimators_values_Slider = [10]
     
    ood_experiment = IForestOODExperiment(machine_type, train_ids, test_id, best_max_features_values_Fan, best_n_estimators_values_Fan)
    ood_experiment.run()