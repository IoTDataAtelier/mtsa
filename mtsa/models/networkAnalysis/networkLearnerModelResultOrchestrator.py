import os
import fastavro

class NetworkLearnerModelResultOrchestrator:
    @staticmethod
    def save_result(network, metricResult, networkName, avro_file_name):
        schema = {
            "type": "record",
            "name": "GANF_Network",
            "fields": [
                {"name": "adj_matrix_name", "type": "string"},
                {"name": "adjacent_matrix", "type": {"type": "array", "items": {"type": "array", "items": "double"}}},
                {"name": "AUC_ROC", "type": "float"},
            ],
        }

        data = [{
            "adj_matrix_name": networkName,  
            "adjacent_matrix": network,    
            "AUC_ROC": float(metricResult),  
        }]
        
        os.makedirs(os.path.dirname(avro_file_name), exist_ok=True)
        
        if os.path.isfile(avro_file_name):
            with open(avro_file_name, "a+b") as f_out:
                fastavro.writer(f_out, schema, data, codec="deflate")
        else:    
            with open(avro_file_name, "wb") as f_out:
                fastavro.writer(f_out, schema, data, codec="deflate")
