import fastavro
import os
import json
import networkx as nx
import numpy as np

class NetworkLearnerObserver():
    def __init__(self, **kwargs) -> None:
        self.avro_file_name_with_path= kwargs["avro_path"]
        self.avro_file_name = ""
        self.json_file_name = ""
        self.network = None
        self.data = []
        self.all_adjacentMatrix = []
        self.latest_args = None 
        
    def update(self, args):
        self.__create_avro_file(args)

    def __create_avro_file(self, args):
        schema = args["schema_avro"]
        avro_file_name = self.avro_file_name_with_path
        #json_file_name = args["file_name"]+".json"
        
        data = args["data"]
        os.makedirs(os.path.dirname(avro_file_name), exist_ok=True)
        
        if os.path.isfile(avro_file_name):
            with open(avro_file_name, "a+b") as f_out:
                fastavro.writer(f_out, schema, data,codec="deflate")
        else:    
            with open(avro_file_name, "wb") as f_out:
                fastavro.writer(f_out, schema, data,codec="deflate")
                
        self.latest_args = args
        #self.__create_json_from_avro(avro_file_name, json_file_name)
        #self.create_network(data)
        #self.data = data
        #self.all_adjacentMatrix.append(data[0]["adjacent_matrix"])

    def __create_json_from_avro(self, avro_file_name, json_file_name):
        with open(avro_file_name, "rb") as avro_file:
            reader = fastavro.reader(avro_file)
            
            information = [info for info in reader]
        
        with open(json_file_name, "w") as json_file:
            json.dump(information, json_file, indent=4)
   
    def create_network(self, data):
        network = nx.Graph()
        adjacent_matrix = np.array(data[0]["adjacent_matrix"])
        nodes = np.arange(0, adjacent_matrix.shape[0]).tolist() 
        network.add_nodes_from(nodes)
        for fixed_node in range(adjacent_matrix.shape[0]):
            for relation_node in range(adjacent_matrix.shape[0]):
                network.add_weighted_edges_from([(fixed_node, relation_node,adjacent_matrix[fixed_node][relation_node])])
                
        self.network = network
        
    def add_AUC_to_avro_file(self, value):
        schema = self.latest_args["schema_avro"]
        avro_file_name = self.avro_file_name_with_path
        data =  self.latest_args["data"]
        data[0]['AUC_ROC'] = value
        
        if os.path.isfile(avro_file_name):
            with open(avro_file_name, "a+b") as f_out:
                fastavro.writer(f_out, schema, data,codec="deflate")
        else:    
            with open(avro_file_name, "wb") as f_out:
                fastavro.writer(f_out, schema, data,codec="deflate")
        

                     
        
        