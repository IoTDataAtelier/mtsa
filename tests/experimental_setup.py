from mtsa import (
    MFCCMix,
    Hitachi,
    FEATURES,
    files_train_test_split,
    calculate_aucroc,
    get_tsne_results,
    files_train_test_split_combined
)
from sklearn.model_selection import LeaveOneOut
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from functools import reduce
import itertools as ite
import numpy as np
import argparse
import os
import json
from scripts.common import elapsed_time, multiple_runs
import time
import functools as ft
from scripts.experiments import *


class PaperScript2023:

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

    def get_features(self):
        number_features = np.arange(1, len(FEATURES)+1)
        def combinations(r): return ite.combinations(FEATURES, r)
        all_combinations = map(combinations, number_features)
        features = reduce(lambda x, y: ite.chain(x, y), all_combinations)
        return features

    def get_models_mfccmix(self):
        def build_mfccmix_by_features(features):
            mfccmix = MFCCMix(
                features=list(features),
                final_model=GaussianMixture(n_components=self.n_components)
            )
            return ModelParam("MFCCMix " + "+".join([f[0] for f in features]), mfccmix)
        features = self.get_features()
        mfcc_models = map(build_mfccmix_by_features, features)
        return mfcc_models

    def get_models_baseline(self):
        hitachi = [ModelParam("Baseline", Hitachi())]
        return hitachi

    def get_models(self):
        model_baseline = self.get_models_baseline()
        models_mfccmix = self.get_models_mfccmix()
        all_models = reduce(lambda x, y: ite.chain(
            x, y), (model_baseline, models_mfccmix))
        return list(all_models)

    def get_paths(self, level=None):
        paths = glob.glob(os.path.join(self.path, f'*{os.sep}' * level))
        return paths
    
    def get_params(self, level=None):
        if not level:
            level = self.level
        all_models = self.get_models()
        all_paths = self.get_paths(level)
        all_params = list(ite.product(all_paths, all_models))
        all_params = [Params(*p) for p in all_params]
        return all_params
                
    def write_json(self, result : ParamResult, **kwargs):
        path_output = os.path.join(self.output, 'rocs')
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        filename = 'filename'
        if 'filename' in kwargs:
            filename = kwargs['filename']
        filepath = os.path.join(path_output, f'{filename}')
        
        ts = time.time_ns()
        filepath = f'{filepath}_{ts}'
        with open(filepath, "a") as jf:
            #TODO this is ugly but leting keras to be serialized causes circular reference at json.dumps 
            should_dump = lambda o: any(map(ft.partial(isinstance, o), [t for t in [int, float, str, ParamResult, Result, Params, ModelParam]]))
            def default(o):
                if should_dump(o):
                    if hasattr(o, "__dict__"):
                        return o.__dict__
                    else:
                        return str(o)
                else:
                    return ""
            json_data = json.dumps(result, indent=4, default=default)
            print(json_data, file=jf)

    def fit_individual_model(self, params: Params):
        # print(params)
        path = params.path
        model = params.model.obj
        model_name = params.model.name
        X_train, X_test, y_train, y_test = files_train_test_split(path)
        result_fit = elapsed_time(model.fit, X_train, y_train)
        param_result_fit = ParamResult(params, result_fit)
        self.write_json(param_result_fit, filename='results_individual_fit')
        result_roc = elapsed_time(calculate_aucroc, model, X_test, y_test)
        param_result_roc = ParamResult(params,result_roc)
        self.write_json(param_result_roc, filename='results_individual_roc')
        x, y, hue = self.local_get_tsne_results(model, X_test, y_test)
        self.plot_tsne(model_name, path, x, y, hue)

    def local_get_tsne_results(self, model, X_test, y_test):
        x, y, hue = get_tsne_results(
            model=model,
            X=model.transform(X_test),
            y=y_test,
            perplexity=self.perplexity
        )
        return  x, y, hue
    
    def plot_tsne(self, model_name, path, x, y, hue):
        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=x, y=y,
            hue=hue,
            palette=sns.color_palette("hls", 2),
            legend="full",
            s=100
            # alpha=0.3
        )
        path_output = os.path.join(self.output, 'results_tse')
        if not os.path.exists(path_output):
            os.makedirs(path_output)
        filename = '_'.join(path.split(os.path.sep)[:-1][-(self.level+1):]) + "_" + model_name
        ts = time.time_ns()
        filename = f'{filename}_{ts}'
        plt.savefig(os.path.join(path_output, f'{filename}.pdf'))

   

    def results_individual(self):
        print("START: results_individual")
        all_params = self.get_params(self.level)
        runs = self.runs
        results = [multiple_runs(runs, self.fit_individual_model, param) for param in all_params]        
        print("FINISH: results_individual")



    def calculate_roc(self, params_tuple):
        paths, model_name, model, split = params_tuple
        path_train = paths[split[0]]
        path_test = paths[split[1]]
        print(
            f"results_combined calculate_roc {str(path_test)} {model_name}")
        train_X_train, train_X_test, train_y_train, train_y_test = files_train_test_split_combined(
            path_train)
        test_X_train, test_X_test, test_y_train, test_y_test = files_train_test_split_combined(
            path_test)
        params = Params(path_test[0], ModelParam(model_name, model))
        result_fit = elapsed_time(model.fit, train_X_train, train_y_train)
        param_result_fit = ParamResult(params, result_fit)
        self.write_json(param_result_fit, filename='results_combined_fit')
        result_roc = elapsed_time(calculate_aucroc, model, test_X_test, test_y_test)
        param_result_roc = ParamResult(params,result_roc)
        self.write_json(param_result_roc, filename='results_combined_roc')
        x, y, hue = self.local_get_tsne_results(model, test_X_test, test_y_test)
        self.plot_tsne(model_name, params.path, x, y, hue)

    def calculate_roc_combined(self, params: Params):
        path = params.path
        model = params.model.obj
        model_name = params.model.name
        # print(
        #     f"results_combined calculate_roc_combined {path} {model_name}")
        paths = np.array(glob.glob(os.path.join(path, f'*{os.sep}')))
        loo = LeaveOneOut()
        splits = loo.split(paths)
        rocs = map(self.calculate_roc, [
                (paths, model_name, model, split) for split in splits])
        return list(rocs)

    # def results_combined(self):
    #     print("START: results_combined")
        
    #     all_params = self.get_params(self.level -1 )

    #     results = [
    #         multiple_runs(self.runs, 
    #         self.calculate_roc_combined, 
    #         param) 
    #         for param in all_params
    #         ]   
        
    #     print("FINISH: results_combined")


# def main(parse_args):
#     paper_script_2023 = PaperScript2023(**parse_args.__dict__)
#     # paper_script_2023.results()


# if __name__ == "__main__":
#     print("MTSA Pipeline")
#     print("Initializing...")
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--type', type=str)
#     parser.add_argument('--sampling_rate', type=int)
#     parser.add_argument('--path', type=str)
#     parser.add_argument('--n_components', type=int)
#     parser.add_argument('--level', type=int)
#     parser.add_argument('--perplexity', type=int)
#     parser.add_argument('--runs', type=int)
#     parser.add_argument('--output', type=str)
#     parse_args = parser.parse_args()
#     main(parse_args)
#     print("MTSA Pipeline: Finished")

# def convert(match_obj):
#             if match_obj.group(0) == '_':
#                 return ""
#             if match_obj.group(0) == '/':
#                 return " "
