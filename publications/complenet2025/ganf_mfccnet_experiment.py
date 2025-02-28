# official-01/08/2024
import os
import sys

module_path = os.path.abspath(os.path.join("../mtsa/"))
if module_path not in sys.path:
    sys.path.append(module_path)

import tensorflow as tf # noqa: E402
import torch # noqa: E402
import numpy as np # noqa: E402
from sklearn.model_selection import KFold # noqa: E402
from multiprocessing import Process, set_start_method # noqa: E402
from mtsa.models.interfaces.observer import Observer # noqa: E402
from mtsa.metrics import calculate_aucroc # noqa: E402
from mtsa.models.ganf import GANF # noqa: E402
from mtsa.utils import files_train_test_split  # noqa: E402


def run_ganf_experiment(data_path):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    batch_size = 128
    learning_rate = 1e-6
    sampling_rate_sound = 16000

    for experiment_index in range(7):
        print("vai até o 7: " + str(experiment_index))
        experiment_path = "./publications/complenet2024/artifacts/exp_" + str(experiment_index) + "/fold_0"
        
        X_train, X_test, Y_train, Y_test = files_train_test_split(data_path)
        
        kf = KFold(n_splits=5)
        dataset_splits = list(enumerate(kf.split(X_train, Y_train)))

        for fold, (train_index, test_index) in dataset_splits:
                                 
            experiment_path_current = experiment_path + str(fold + 1) + ".avro"
            x_train_fold, y_train_fold = (
                X_train[train_index],
                Y_train[train_index],
            )
             
            observer = Observer(avro_path=str(experiment_path_current))
            
            model_GANF = GANF(
                sampling_rate=sampling_rate_sound,
                mono=True,
                use_array2mfcc=True,
                isForWaveData=True,
            )
            
            model_GANF.final_model.attach_observer(observer)
             
            model_GANF.fit(
                x_train_fold,
                y_train_fold,
                batch_size=int(batch_size),
                learning_rate=learning_rate,
            )
             
            auc = calculate_aucroc(model_GANF, X_test, Y_test)
            observer.add_AUC_to_avro_file(auc)
                            
            del model_GANF


def info(title):
    print(title)
    print("module name:", __name__)
    print("parent process:", os.getppid())
    print("process id:", os.getpid())


with torch.cuda.device(1):
    if __name__ == "__main__":
        set_start_method("spawn")
        info("main line")
        path_input = ["/data/MIMII/slider/id_00"]
        p = Process(target=run_ganf_experiment, args=(path_input))
        p.start()
        p.join()
