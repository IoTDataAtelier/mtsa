import os
import pandas as pd
import numpy as np
import re
from scipy import stats

class ExperimentAnalyzer:
    def __init__(self, base_dir):
        """
        Inicializa o ExperimentAnalyzer com o diretório base dos experimentos.
        
        Parâmetros:
            - base_dir (str): Caminho para o diretório base onde estão armazenados os resultados dos experimentos.
        """
        self.base_dir = base_dir
        
    def extract_machine_info(self, subfolder_name):
        """
        Extrai o tipo de máquina e o ID a partir do nome da subpasta.
        
        Exemplo: 'FAN_00' -> ('fan', '00')
        
        Parâmetros:
            - subfolder_name (str): Nome da subpasta.
        
        Retorna:
            - (str, str): Tipo de máquina e ID.
        """
        match = re.match(r'^(fan|pump|slider|valve)_(\d+)$', subfolder_name.lower())
        if match:
            return match.group(1), match.group(2)
        else:
            return 'unknown', 'unknown'
    
    def calculate_confidence_interval(self, data, confidence=0.95, bootstrap_samples=1000):
        """
        Calcula a média e os intervalos de confiança utilizando bootstrap.
        
        Parâmetros:
            - data (array-like): Dados para calcular as estatísticas.
            - confidence (float): Nível de confiança desejado (default: 0.95).
            - bootstrap_samples (int): Número de amostras de bootstrap (default: 1000).
        
        Retorna:
            - (float, float, float): Média, limite inferior e limite superior do intervalo de confiança.
        """
        # Calcula a média dos dados originais
        mean = np.mean(data)
        
        # Realiza o bootstrap: reamostragem dos dados 1000 vezes com reposição
        bootstrapped_means = [
            np.mean(np.random.choice(data, size=len(data), replace=True))
            for _ in range(bootstrap_samples)
        ]
        
        # Calcula os limites inferior e superior com base no percentil
        lower_bound = np.percentile(bootstrapped_means, (1 - confidence) / 2 * 100)
        upper_bound = np.percentile(bootstrapped_means, (1 + confidence) / 2 * 100)
        
        return mean, lower_bound, upper_bound
    
    def load_data(self):
        """
        Carrega os CSV a partir das subpastas.
        
        Retorna:
            - pandas.DataFrame: DataFrame combinado com todos os dados carregados.
        """
        subfolders = [f.path for f in os.scandir(self.base_dir) if f.is_dir()]
        
        df_list = []
        
        for subfolder in subfolders:
            subfolder_name = os.path.basename(subfolder)
            machine_type, machine_id = self.extract_machine_info(subfolder_name)
            
            if machine_type == 'unknown':
                print(f"Ignorando a subpasta desconhecida: {subfolder_name}")
                continue
            
            csv_files = [file for file in os.listdir(subfolder) if file.endswith('.csv')]
            
            if not csv_files:
                print(f"Nenhum arquivo CSV encontrado na subpasta: {subfolder_name}")
                continue
            
            for file in csv_files:
                file_path = os.path.join(subfolder, file)
                try:
                    df = pd.read_csv(file_path)
                    df['machine_type'] = machine_type
                    df['machine_id'] = machine_id
                    df_list.append(df)
                except Exception as e:
                    print(f"Erro ao ler o arquivo {file_path}: {e}")
                    continue
            
        if not df_list:
            raise ValueError("Nenhum dado foi carregado. Verifique os arquivos CSV e a estrutura das pastas.")
        
        df_all = pd.concat(df_list, ignore_index=True)
        
        return df_all
    
    def calculate_statistics(self, df_all, group_by_params):
        """
        Calcula a média e os intervalos de confiança das AUC ROCs para cada combinação de parâmetros.
        
        Parâmetros:
            - df_all (pandas.DataFrame): DataFrame combinado com todos os dados carregados.
            - group_by_params (list): Lista de nomes de colunas para agrupar os dados.
        
        Retorna:
            - pandas.DataFrame: DataFrame com as estatísticas calculadas.
        """
        grouping_columns = ['machine_type', 'machine_id'] + group_by_params
        df_stats = df_all.groupby(grouping_columns).apply(
            lambda x: pd.Series({
                'Mean_AUC_ROC': np.mean(x['AUC_ROC']),
                'CI_lower': self.calculate_confidence_interval(x['AUC_ROC'])[1],
                'CI_upper': self.calculate_confidence_interval(x['AUC_ROC'])[2]
            })
        ).reset_index()
    
        return df_stats
    
    def save_statistics(self, df_stats, group_by_params):
        """
        Salva as estatísticas calculadas em novos arquivos CSV.
        
        Parâmetros:
            - df_stats (pandas.DataFrame): DataFrame com as estatísticas calculadas.
            - group_by_params (list): Lista de nomes de parâmetros usados para agrupar (usado nos nomes dos arquivos).
        """
        for _, row in df_stats.iterrows():
            machine_type = row['machine_type']
            machine_id = row['machine_id']
            
            # Define a pasta de destino para salvar as estatísticas
            dest_dir = os.path.join(self.base_dir, f'confidence_interval_{machine_type.upper()}_{machine_id}')
            os.makedirs(dest_dir, exist_ok=True)
            
            # Define o nome do arquivo de estatísticas dinamicamente
            param_strings = [f"{param}-{row[param]}" for param in group_by_params]
            stat_filename = f"statistics_{'_'.join(param_strings)}.csv"
            stat_filepath = os.path.join(dest_dir, stat_filename)
            
            # Cria um DataFrame com os dados a serem salvos
            columns_to_save = ['machine_type', 'machine_id'] + group_by_params + ['Mean_AUC_ROC', 'CI_lower', 'CI_upper']
            stat_df = pd.DataFrame({col: [row[col]] for col in columns_to_save})
            
            # Salva o DataFrame em CSV
            stat_df.to_csv(stat_filepath, index=False)
            print(f"Estatísticas salvas em: {stat_filepath}")
    
    def run_analysis(self, group_by_params):
        """
        Executa todo o fluxo de análise: carregamento de dados, cálculo de estatísticas e salvamento.
        
        Parâmetros:
            - group_by_params (list): Lista de nomes de colunas para agrupar os dados.
        """
        df_all = self.load_data()
        
        df_stats = self.calculate_statistics(df_all, group_by_params)
        
        self.save_statistics(df_stats, group_by_params)
        
        print("Análise completa concluída.")
