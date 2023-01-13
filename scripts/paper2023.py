'''

'''
#Disable Logging
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.keras.utils.disable_interactive_logging()

import argparse
import os
import numpy as np
import itertools as ite
import pandas as pd
from functools import reduce
import glob
import matplotlib.pyplot as plt
from mtsa import (
    MFCCMix,
    Hitachi,
    FEATURES,
    files_train_test_split,
    calculate_aucroc,
    get_tsne_results,
    files_train_test_split_combined
)
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneOut


class PaperScript2023:

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


    def results(self):
        self.results_01_individual()
        self.results_02_tse()
        self.results_03_combined()
   
    def get_features(self):
        number_features = np.arange(1, len(FEATURES)+1)
        combinations = lambda r: ite.combinations(FEATURES, r)
        all_combinations = map(combinations, number_features)
        features = reduce(lambda x,y: ite.chain(x,y), all_combinations)
        return features
        
    def get_models_mfccmix(self):
        def build_mfccmix_by_features(features):
            mfccmix = MFCCMix(
                features=list(features),
                final_model=GaussianMixture(n_components=self.n_components)
            )
            return ("MFCCMix " + "+".join([f[0] for f in features]), mfccmix)
        features = self.get_features()
        mfcc_models = map(build_mfccmix_by_features, features)
        return mfcc_models
        
    def get_models_baseline(self):
        hitachi = [("Baseline", Hitachi())]
        return hitachi
    
    def get_models(self):
        model_baseline = self.get_models_baseline()
        models_mfccmix = self.get_models_mfccmix()
        all_models = reduce(lambda x,y: ite.chain(x,y), (model_baseline, models_mfccmix))
        return list(all_models)

    def get_paths(self, level):
        paths = glob.glob(os.path.join(self.path, f'*{os.sep}' * level))
        return paths

    def write_df(self, df_final, **kwargs):
        path_output = os.path.join(self.output, 'rocs')
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        filename = 'roc'
        if 'filename' in kwargs:
            filename = kwargs['filename']
        with open(os.path.join(path_output, f'{filename}.tex'), 'w') as tf:
            tf.write(df_final.to_latex())     
    
    def convert_to_dataframe(self, rocs, level):
        rocs_norms = list(map(self.extract_params, [(r, level) for r in rocs]))
        # features = list(self.get_paths(level))]
        df = pd.DataFrame(rocs_norms)
        col_params = [f'P{p}' for p in range(level+1)]
        col_features = ['features', 'roc']
        df_params = pd.DataFrame(df[0].tolist(), index=df.index, columns=col_params)
        df_rocs = pd.DataFrame(df[[1,2]])
        df_rocs.columns = col_features
        df_final = pd.concat([df_params, df_rocs], axis=1)
        df_final = pd.pivot(df_final, index=list(df_params.columns), columns='features', values='roc')
        df_final = df_final[sorted(df_final.columns, key=len)]
        df_final = df_final.loc[df_final.index].apply(lambda x: round(x, 2))
        return df_final

    def extract_params(self, params):
        roc, level = params
        params = roc[0].split(os.sep)[:-1][-(level+1):]
        return tuple(params), roc[1], roc[2]

    def results_01_individual(self):
        print("START: results_01_individual")
        def calculate_roc(params):
            path, model = params
            model_name, model = model
            print(f"results_01_individual calculate_roc {path} {model_name}")
            X_train, X_test, y_train, y_test = files_train_test_split(path)
            model.fit(X_train, y_train)
            roc = calculate_aucroc(model,X_test, y_test)
            return path, model_name, roc
            
        all_models = self.get_models()
        all_paths = self.get_paths(self.level)
        all_params = list(ite.product(all_paths, all_models))
        rocs = list(map(calculate_roc, all_params))
        df_final = self.convert_to_dataframe(rocs, level=self.level)
        self.write_df(df_final, filename='results_01_individual')
        print("FINISH: results_01_individual")

    def results_02_tse(self):
        print("START: results_02_tse")
        
        def local_get_tsne_results(params):
            path, model = params
            model_name, model = model
            print(f"results_02_tse local_get_tsne_results {path} {model_name}")
            X_train, X_test, y_train, y_test = files_train_test_split(path)
            x, y, hue = get_tsne_results(
                model=model,
                X=model.transform(X_test), 
                y=y_test, 
                perplexity=self.perplexity
                )   
            return path, model_name, x, y, hue

        def plot_tsne(params):
            path, model_name, x, y, hue = params
            plt.figure(figsize=(16,10))
            sns.scatterplot(
                x=x, y=y,
                hue=hue,
                palette=sns.color_palette("hls", 2),
                legend="full",
                s=100
                # alpha=0.3
                )
            path_output = os.path.join(self.output, 'results_02_tse')
            if not os.path.exists(path_output):
                os.makedirs(path_output)
            filename = '_'.join(path.split(os.path.sep)[:-1][-(self.level+1):]) + "_" + model_name
            plt.savefig(os.path.join(path_output, f'{filename}.pdf'))
            
        all_models = self.get_models()
        all_paths = self.get_paths(self.level)
        all_params = list(ite.product(all_paths, all_models))
        tsne_results = list(map(local_get_tsne_results, all_params))
        list(map(plot_tsne, tsne_results))
        print("FINISH: results_02_tse")

    def results_03_combined(self):
            print("START: results_03_combined")

            def calculate_roc(params):
                paths, model_name, model, split = params
                train = paths[split[0]]
                test = paths[split[1]]
                print(f"results_03_combined calculate_roc {str(test)} {model_name}")
                train_X_train, train_X_test, train_y_train, train_y_test = files_train_test_split_combined(train)
                test_X_train, test_X_test, test_y_train, test_y_test = files_train_test_split_combined(test)
                model.fit(train_X_train, train_y_train)
                roc = calculate_aucroc(model, test_X_test, test_y_test)
                return str(test[0]), model_name, roc
            
            def calculate_roc_combined(params):
                
                path, model_params = params
                model_name, model = model_params
                print(f"results_03_combined calculate_roc_combined {path} {model_name}")
                paths = np.array(glob.glob(os.path.join(all_paths[0], f'*{os.sep}' )))
                loo = LeaveOneOut()
                splits = loo.split(paths) 
                rocs = map(calculate_roc, [(paths, model_name, model, split) for split in splits])
                return rocs

            all_paths = self.get_paths(self.level - 1)
            all_models = self.get_models()
            all_params = list(ite.product(all_paths, all_models))
            rocs = list(map(calculate_roc_combined, all_params))
            rocs = reduce(lambda x,y: ite.chain(x,y), rocs)
            rocs = list(rocs)
            df_final = self.convert_to_dataframe(rocs, level=self.level -1)
            self.write_df(df_final, filename='results_03_combined')
            print("FINISH: results_03_combined")

def main(parse_args):
    paper_script_2023 = PaperScript2023(**parse_args.__dict__)
    paper_script_2023.results()
        
if __name__ == "__main__":
    print("MTSA Pipeline")
    print("Initializing...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_rate',type=int)
    parser.add_argument('--path', type=str)
    parser.add_argument('--n_components', type=int)
    parser.add_argument('--level', type=int)
    parser.add_argument('--perplexity', type=int)
    parser.add_argument('--output', type=str)
    parse_args = parser.parse_args()
    main(parse_args)    
    print("MTSA Pipeline: Finished")
