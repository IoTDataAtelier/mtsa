import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
tf.keras.utils.disable_interactive_logging()

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
config = tf.config.experimental.set_memory_growth(physical_devices[1], True)


from apache_beam.io.textio import WriteToText
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneOut
import pyarrow as pa
import numpy as np
import itertools as ite
from functools import reduce
from mtsa import (
    MFCCMix,
    Hitachi,
    FEATURES,
    files_train_test_split,
    calculate_aucroc,
    get_tsne_results,
    files_train_test_split_combined
)
import os
import glob
from functools import partial
from mtsa.metrics import calculate_aucroc
from mtsa.common import elapsed_time
from mtsa.utils import files_train_test_split
from apache_beam.io import WriteToText
from apache_beam.pipeline import PipelineOptions
import apache_beam as beam
import argparse

from mtsa import get_tsne_results


class IndividualCreateKeyFn(beam.DoFn):
    def __init__(self, *args, **kwargs) -> None:
        beam.DoFn.__init__(self)
        self.__dict__.update(kwargs)

    def process(self, args):
        model_name, model = args
        all_paths = glob.glob(os.path.join(self.path, f'*{os.sep}' * self.level))
        for path in all_paths:
            X_train, X_test, y_train, y_test = files_train_test_split(path)
            metrics = []
            yield (path, model_name, model, X_train, X_test, y_train, y_test, metrics)

class CombinedCreateKeyFn(beam.DoFn):
    def __init__(self, *args, **kwargs) -> None:
        beam.DoFn.__init__(self)
        self.__dict__.update(kwargs)

    def process(self, args):
        model_name, model = args
        all_paths = glob.glob(os.path.join(self.path, f'*{os.sep}' * (self.level - 1)))
        for path in all_paths:
            paths = np.array(glob.glob(os.path.join(path, f'*{os.sep}')))
            loo = LeaveOneOut()
            splits = loo.split(paths)
            for split in splits:
                path_train = paths[split[0]]
                path_test = paths[split[1]]
                train_X_train, train_X_test, train_y_train, train_y_test = files_train_test_split_combined(path_train)
                test_X_train, test_X_test, test_y_train, test_y_test = files_train_test_split_combined(path_test)
                metrics = []
                yield (path_test[0], model_name, model, train_X_train, test_X_test, train_y_train, test_y_test, metrics)

class FitFn(beam.DoFn):
    def __init__(self, *args, **kwargs) -> None:
        beam.DoFn.__init__(self)
        self.__dict__.update(kwargs)

    def process(self, element):
        path, model_name, model, X_train, X_test, y_train, y_test, metrics = element
        result_fit = elapsed_time(model.fit, X_train, y_train)
        metrics.append({
            "name": "fit_time",
            "value": result_fit.time_elapsed
        }
        )
        yield (path, model_name, model, X_train, X_test, y_train, y_test, metrics)

class RocFn(beam.DoFn):
    def __init__(self, *args, **kwargs) -> None:
        beam.DoFn.__init__(self)
        self.__dict__.update(kwargs)

    def process(self, element):
        path, model_name, model, X_train, X_test, y_train, y_test, metrics = element
        result_roc = elapsed_time(
            calculate_aucroc, model, X_test, y_test)
        metrics.append({
            "name": "roc_time",
            "value": result_roc.time_elapsed
        },)
        metrics.append({
            "name": "roc_value",
            "value": result_roc.fun_return
        },)
        yield (path, model_name, model, X_train, X_test, y_train, y_test, metrics)


class TSNEFn(beam.DoFn):
    def __init__(self, *args, **kwargs) -> None:
        beam.DoFn.__init__(self)
        self.__dict__.update(kwargs)

    def process(self, element):
        path, model_name, model, X_train, X_test, y_train, y_test, metrics = element
        x, y, hue = get_tsne_results(model, X_test, y_test)
        tsne = {
            'x': list(x),
            'y': list(y),
            'hue': list(hue)
        }
        yield (path, model_name, model, X_train, X_test, y_train, y_test, metrics, tsne)


class TransformFn(beam.DoFn):
    def __init__(self, *args, **kwargs) -> None:
        beam.DoFn.__init__(self)
        self.__dict__.update(kwargs)

    def process(self, element):
        path, model_name, model, X_train, X_test, y_train, y_test, metrics, tsnes = element

        yield {
            "path": path,
            "model": model_name,
            "metrics": metrics,
            "tsnes": tsnes
        }

def get_models_mfccmix(n_components):
    def get_features():
        number_features = np.arange(1, len(FEATURES)+1)
        def combinations(r): return ite.combinations(FEATURES, r)
        all_combinations = map(combinations, number_features)
        features = reduce(lambda x, y: ite.chain(x, y), all_combinations)
        return features

    def build_mfccmix_by_features(features):
        mfccmix = MFCCMix(
            features=list(features),
            final_model=GaussianMixture(n_components=n_components)
        )
        return "MFCCMix " + "+".join([f[0] for f in features]), mfccmix
    
    features = get_features()
    mfcc_models = map(build_mfccmix_by_features, features)
    return mfcc_models


def get_models(n_components):
    models_hitachi = [("Hitachi", Hitachi())]
    # models_mfccmix = get_models_mfccmix(n_components)
    models_mfccmix = []
    all_models = reduce(lambda x, y: ite.chain(
        x, y), (models_hitachi, models_mfccmix))
    return list(all_models)


def run_pipeline(known_args):

    if known_args.script == 'individual':
        create_key_fn = IndividualCreateKeyFn
    elif known_args.script == 'combined':
        create_key_fn = CombinedCreateKeyFn

    with beam.Pipeline() as pipeline:

        models = pipeline | "models" >> beam.Create(known_args.runs * get_models(known_args.n_components))

        keys = models | "keys" >> beam.ParDo(create_key_fn(path=known_args.path, level=known_args.level))

        fits = keys | "train" >> beam.ParDo(FitFn())

        rocs = fits | "test" >> beam.ParDo(RocFn())
        
        tsnes = rocs | "tsne" >> beam.ParDo(TSNEFn())

        transform = tsnes | beam.ParDo(TransformFn(path=known_args.path))

        write = transform | WriteToText(
            f"{known_args.output}{known_args.script}.text")


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('--script', choices=['individual', 'combined'], required=True)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--runs', type=int, required=True)
    parser.add_argument('--n_components', type=int, required=True)
    parser.add_argument('--sampling_rate', type=int)
    # parser.add_argument('--perplexity', type=int)

    known_args, pipeline_args = parser.parse_known_args(argv)

    beam_options = PipelineOptions()

    run_pipeline(known_args)

if __name__ == "__main__":
    main()

