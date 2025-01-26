import os
import fastavro

class MetricNetworkResult:
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

        ## Verifique se 'network' está no formato correto
        #if not isinstance(network, list) or not all(isinstance(row, list) for row in network):
        #    raise ValueError("O parâmetro 'network' deve ser uma lista de listas.")

        ## Verifique se 'metricResult' é um número (float)
        #if not isinstance(metricResult, (float, int)):
        #    raise ValueError("O parâmetro 'metricResult' deve ser um número (float ou int).")

        # Dados a serem salvos
        data = [{
            "adj_matrix_name": networkName,  
            "adjacent_matrix": network,    
            "AUC_ROC": float(metricResult),  # Certifique-se de que é um float
        }]
        
        os.makedirs(os.path.dirname(avro_file_name), exist_ok=True)
        
        if os.path.isfile(avro_file_name):
            with open(avro_file_name, "a+b") as f_out:
                fastavro.writer(f_out, schema, data, codec="deflate")
        else:    
            with open(avro_file_name, "wb") as f_out:
                fastavro.writer(f_out, schema, data, codec="deflate")
