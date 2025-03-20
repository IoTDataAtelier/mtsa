import numpy as np

class NetworkLearnerModel:
    def __init__(self):
        super().__init__()
        self.observers = []

    def attach_observer(self, observer):
        self.observers.append(observer)

    def dettach_observer(self, observer):
        self.observers.remove(observer)

    def notify_observers(self, adjacent_matrix, epoch, h, total_loss, adj_matrix_name, learning_rate, batch_size):
        self.__proxy_notify_observers(
            current_epoch=epoch,
            adjacent_matrix=adjacent_matrix.clone(),
            loss=total_loss.clone(),
            DAG_constraint_h=h.clone(),
            models_specs={
                "learning_rate:": learning_rate,
                "batch_size": batch_size,
            },
            AUC_ROC=-np.inf,
            adj_matrix_name=adj_matrix_name,
        )
        
    def __notifyNetworkExtractor(self, **kwargs):
        for observer in self.observers:
            observer.update(kwargs)

    def __proxy_notify_observers(self, **kwargs):
        schema_avro = {
            "type": "record",
            "name": "GANF_Network",
            "fields": [
                {"name": "adj_matrix_name", "type": "string"},
                {"name": "current_epoch", "type": "int"},
                {"name": "loss", "type": "float"},
                {"name": "DAG_constraint_h", "type": "float"},
                {
                    "name": "adjacent_matrix",
                    "type": {
                        "type": "array",
                        "items": {"type": "array", "items": "double"},
                    },
                },
                {"name": "models_specs", "type": {"type": "map", "values": "float"}},
                {"name": "AUC_ROC", "type": "float"},
            ],
        }
        data = [
            {
                "current_epoch": kwargs["current_epoch"],
                "loss": kwargs["loss"],
                "DAG_constraint_h": kwargs["DAG_constraint_h"],
                "adjacent_matrix": kwargs["adjacent_matrix"].tolist(),
                "models_specs": kwargs["models_specs"],
                "AUC_ROC": kwargs["AUC_ROC"],
                "adj_matrix_name": kwargs["adj_matrix_name"],
            }
        ]

        self.__notifyNetworkExtractor(schema_avro=schema_avro, data=data)
