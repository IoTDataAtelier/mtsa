# official-01/08/2024
import os
import sys
import warnings


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
    batch_size = 128
    learning_rate = 1e-6
    sampling_rate_sound = 16000
    epochs = 50
    X_train, X_test, Y_train, Y_test = files_train_test_split(data_path)
                             
    experiment_path_current = experiment_path 
    
    print(experiment_path_current)
    
    observer = Observer(avro_path=str(experiment_path_current))
    model_GANF = GANF(
        sampling_rate=sampling_rate_sound,
        mono=True,
        use_array2mfcc=True,
        isForWaveData=True,
        device=str(cuda_device)
    )
    
    model_GANF.final_model.attach_observer(observer)
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
    observer.add_AUC_to_avro_file(auc)
    
    finalNet = model_GANF.get_adjacent_matrix()
    MetricNetworkResult.save_result(finalNet, auc, "final network", netPath)
    
    initalNet = model_GANF.getInitalAdj()
    auc = calculate_aucroc(model_GANF.set_adjacent_matrix(initalNet), X_test, Y_test)
    MetricNetworkResult.save_result(initalNet,auc, "intial network", netPath)
    
    randomNet = model_GANF.get_random_adjacent_matrix()
    auc = calculate_aucroc(model_GANF.set_adjacent_matrix(randomNet), X_test, Y_test)
    MetricNetworkResult.save_result(randomNet,auc, "intial network", netPath)
                    
    del model_GANF


def configure_gpu(cuda_device):
    # Define o CUDA device a ser utilizado
    tf_gpus = tf.config.experimental.list_logical_devices("GPU")
    print(f"TensorFlow device: {tf_gpus[int(cuda_device)] if tf_gpus else 'CPU'}")
   
    
    

if __name__ == "__main__":
    # Captura os parâmetros da linha de comando
    if len(sys.argv) != 6:
        print("Usage: python experimento.py <path_input> <experiment_path> <netPath> <cuda_device> <session_name>")
        sys.exit(1)

    path_input = sys.argv[1]
    experiment_path = sys.argv[2]
    netPath = sys.argv[3]
    cuda_device = str(sys.argv[4])
    session_name = sys.argv[5]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    import tensorflow as tf
    import torch 
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    print("cuda_device: "+ cuda_device)
    with torch.cuda.device(int(cuda_device)):
    #configure_gpu(cuda_device)
        run_ganf_experiment(path_input, experiment_path, netPath, cuda_device, session_name)
