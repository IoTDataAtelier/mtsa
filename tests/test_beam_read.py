from apache_beam.io import ReadFromAvro
import apache_beam as beam
from apache_beam.pipeline import PipelineOptions
# from apache_beam.dataframe.convert import to_dataframe
# from scripts.analysis import get_df_from_path

output = "/data/output_test/"


def main(argv=None):
    
    with beam.Pipeline() as pipeline:

        runs = (
            pipeline 
            | "read" >> ReadFromAvro(f"{output}individual*")
            | "print" >> beam.Map(print)
            # | "set schema" >> beam.Map (
            #     lambda x: beam.Row(
            # path=x['path'], 
            # model=x['model'],
            # fit_time=x['metrics'][0]['value'],
            # roc_time=x['metrics'][1]['value'],
            # roc_value=x['metrics'][2]['value']
            # )
            )
        
        # df = to_dataframe(runs)
        # df.to_csv(f"{output}individual.csv", index=False)


if __name__ == "__main__":
    main()