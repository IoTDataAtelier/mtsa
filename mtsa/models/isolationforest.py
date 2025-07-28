import numpy as np
import pandas as pd
import time
from sklearn.base import BaseEstimator, OutlierMixin, check_array
from sklearn.pipeline import (
    Pipeline, 
    FeatureUnion
) 
from mtsa.features.stats import (
    FEATURES,
    get_features
    )

from mtsa.features.mel import (
    Array2Mfcc 
)
from mtsa.utils import (
    Wav2Array,
)

from sklearn.ensemble import IsolationForest
from functools import reduce

class IForest(BaseEstimator, OutlierMixin):
    """
    Parâmetros:
        n_estimators (int): Número de árvores de isolamento a serem construídas.
        max_samples (int, float, str): Número de amostras a serem usadas para construir cada árvore.
        contamination (float, str): Estimativa de outliers no conjunto de dados.
        max_features (float, int): Número máximo de características a serem consideradas para cada divisão em uma árvore.
        bootstrap (bool): Se True, amostras são extraídas com substituição. Se false, cada árvore usa amostras únicas.
        n_jobs (int): Número de jobs a serem executados em paralelo.
        random_state (int, RandomState instance, None): Semente para gerador de números aleatórios.
        verbose (int): Controle da verbosidade.
        warm_start (bool): Reutiliza a solução do ajuste anterior para adicionar mais estimadores.
        final_model (estimator): Modelo final.
        features (list): Lista de características.
        sampling_rate (int): Taxa de amostragem para processamento de áudio.
    """

    def __init__(self,
                 n_estimators=100,
                 max_samples=256,
                 contamination=0.01,
                 max_features=1.0,
                 bootstrap=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False, 
                 features=FEATURES,
                 sampling_rate=None, 
                 ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.sampling_rate = sampling_rate
        self.features = features
        self.last_fit_time = 0
        self.execution_duration = None
        self.model_parameters_names = None
        self.dataset_name = None
        self.experiment_dataframe = self.__get_initialize_dataframe_experiments_result()
        self.model = self._build_model()

    @property
    def name(self):
        return "IsolationForest" 
        
    def fit(self, X, y=None):        
        start = time.perf_counter()    
        self.model.fit(X, y)
        end = time.perf_counter()
        self.last_fit_time = end - start
        return self

    def transform(self, X, y=None):
        l = list()
        l.append(X)
        l.extend(self.model.steps[:-1])
        Xt = reduce(lambda x, y: y[1].transform(x), l)
        return Xt
    
    def predict(self, X):
        _predict = self.model.predict(X)
        _predict = np.where(_predict == -1, 0, 1)
        return _predict
    
    #def decision_function(self, X):
        """
        Retorna a pontuação de decisão para cada ponto de dados.
        
        Parâmetros:
            - X (array-like): Dados de entrada.
        
        Retorna:
            array-like: Pontuações de decisão.
        """
    #  return self.model.decision_function(X)
    
    def score(self, X, y=None):
        return self.model.score(X)

    # Retorna a pontuação de anomalia para cada ponto de dados
    def score_samples(self, X):
        return self.model.score_samples(X=X)
    
    def __get_initialize_dataframe_experiments_result(self):
        """
        Inicializa o DataFrame para armazenar os resultados dos experimentos.
        
        Retorna:
            pd.DataFrame: DataFrame vazio com as colunas definidas.
        """
        parameters_columns = self.__get_parameters_columns()                              
        return pd.DataFrame(columns=parameters_columns)
    
    def __get_parameters_columns(self):
        """
        Define as colunas do DataFrame de experimentos.
        
        Retorna:
            list: Lista de nomes de colunas.
        """
        parameters_columns = ["actual_dataset",
                              "parameters_names",
                              "n_estimators",
                              "max_samples",
                              "contamination",
                              "max_features",
                              "execution_duration",
                              "AUC_ROC",
                            ]
        return parameters_columns   

    def __create_dataframe(self):
        """
        Adiciona uma nova linha ao DataFrame de experimentos com os parâmetros e resultados atuais.
        """
        self.experiment_dataframe.loc[len(self.experiment_dataframe)] = {
            "actual_dataset": self.dataset_name,
            "parameters_names": self.model_parameters_names,
            "n_estimators": self.n_estimators, 
            "max_samples": self.max_samples, 
            "contamination": self.contamination, 
            "max_features": self.max_features,
            "execution_duration": self.execution_duration,
            "RMSE": None,
            "Score": None,
            "AUC_ROC": None
            } 
    
    def get_experiment_dataframe(self, dataset_name=None, model_parameters_names=None, execution_duration=None):
        self.dataset_name = dataset_name
        self.model_parameters_names = model_parameters_names
        self.__create_dataframe()
        return self.experiment_dataframe

    def _build_model(self):
        # Etapa 1: Converter áudio para array NumPy
        wav2array = Wav2Array(sampling_rate=self.sampling_rate)
        
        # Etapa 2: Converter array de áudio para MFCCs com n_mfcc especificado
        array2mfcc = Array2Mfcc(sampling_rate=self.sampling_rate)
        
        # Etapa 3: Combinar as features existentes usando FeatureUnion (Retorno virá 2D)
        features = FeatureUnion(self.features)

        # Configuração do IsolationForest
        self.final_model = IsolationForest(
            n_estimators=self.n_estimators, 
            max_samples=self.max_samples, 
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs, 
            random_state=self.random_state,
            verbose=self.verbose,
            warm_start=self.warm_start,
        )
        
        # Construção do pipeline
        model = Pipeline(
            steps=[
                ("wav2array", wav2array),
                ("array2mfcc", array2mfcc),
                ("features", features),
                ("final_model", self.final_model),
            ]
        )
        
        return model

