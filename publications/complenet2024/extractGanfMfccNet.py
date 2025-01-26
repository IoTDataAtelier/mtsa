# official-01/08/2024
import os
import warnings
import tensorflow as tf
import torch
import sys
from multiprocessing import Process, set_start_method 

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Nível 3 suprime todos os logs de informação e warnings

module_path = os.path.abspath(os.path.join("../mtsa/"))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from mtsa.models.interfaces.aucRocModelResults import MetricNetworkResult
from mtsa.models.interfaces.observer import Observer # noqa: E402
from mtsa.metrics import calculate_aucroc # noqa: E402
from mtsa.models.ganf import GANF # noqa: E402
from mtsa.utils import files_train_test_split  # noqa: E402
    

def run_ganf_experiment(data_path, experiment_path,netPath,cuda_device,session_name):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    batch_size = 128
    learning_rate = 1e-6
    sampling_rate_sound = 16000
    epochs = 50
    X_train, X_test, Y_train, Y_test = files_train_test_split(data_path)
                             
    experiment_path_current = experiment_path 
    
    print(experiment_path_current)
    
    #observer = Observer(avro_path=str(experiment_path_current))
    model_GANF = GANF(
        sampling_rate=sampling_rate_sound,
        mono=True,
        use_array2mfcc=True,
        isForWaveData=True,
        index_CUDA_device=str(cuda_device)
    )
    
    
    #model_GANF.final_model.attach_observer(observer)
    print("training model...")
     
    model_GANF.fit(
        X_train,
        Y_train,
        batch_size=int(batch_size),
        learning_rate=learning_rate,
        epochs = epochs
    )
    
    print("testing model...")
    auc = calculate_aucroc(model_GANF, X_test, Y_test)
    #observer.add_AUC_to_avro_file(auc)
    
    finalNet = model_GANF.get_adjacent_matrix()
    MetricNetworkResult.save_result(finalNet, auc, "final network", netPath)
    
    initalNet = model_GANF.getInitalAdj()
    model_GANF.set_adjacent_matrix(initalNet)
    auc = calculate_aucroc(model_GANF, X_test, Y_test)
    MetricNetworkResult.save_result(initalNet,auc, "intial network", netPath)
    
    randomNet = model_GANF.get_random_adjacent_matrix()
    model_GANF.set_adjacent_matrix(randomNet)
    auc = calculate_aucroc(model_GANF, X_test, Y_test)
    MetricNetworkResult.save_result(randomNet,auc, "random network", netPath)
                    
    del model_GANF


def info(title):
    print(title)
    print("module name:", __name__)
    print("parent process:", os.getppid())
    print("process id:", os.getpid())

with torch.cuda.device(1):
    if __name__ == "__main__":
            
        i = 9
        path_input = "/data/MIMII/slider/id_00"
        experiment_path=  "../complenet2025/exp05112024/net/exp_" + str(i + 1) + ".avro"
        netPath = "../complenet2025/exp05112024/metric/metr_"+ str(i + 1) + ".avro"
        session_name = f"exp_{i}"
        
        set_start_method("spawn")
        info("main line")
        print("session: " + session_name)
        p = Process(target=run_ganf_experiment, args=(path_input, experiment_path, netPath, 1,session_name))
        p.start()
        p.join()
            


    