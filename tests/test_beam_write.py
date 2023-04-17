from apache_beam.io.avroio import WriteToAvro

import apache_beam as beam
from apache_beam.pipeline import PipelineOptions

output = "/data/output_test/"

schema = {
    "namespace": "mtsa",
    "name": "ExperimentResult", 
    "type": "record",
    "fields": [
        {"name": "len", "type": "int"}
    ]
} 

def main(argv=None):
        
    
    with beam.Pipeline() as pipeline:

        runs = pipeline | beam.Create([
            "Hi",
            "This is a line",
            "To be written"
        ])

        params_individual =  runs | "individual" >> beam.Map(lambda x: {"len": len(x)})

        # write = params_individual | WriteToAvro(f"{output}test")
        write = params_individual | WriteToAvro(f"{output}test.avro", schema=schema)



if __name__ == "__main__":
    main()