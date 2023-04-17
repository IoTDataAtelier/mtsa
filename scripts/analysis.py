import pandas as pd
import glob
from scripts.experiments import *
import json 

def read_data(path):
    files = glob.glob(f'{path}*')
    data = [ParamResult(**json.load(open(f))) for f in files]
    return data




def read_df_from_json(path):
    data_fit = read_data(path = f"{path}_fit")
    data_roc = read_data(path = f"{path}_roc")
    df = pd.DataFrame.from_dict(
                {
                    (param_result[0].params['path'],
                    param_result[0].params['model']['name'],
                    param_result[2]
                    ):
                        {
                            "fit time": param_result[0].result['time_elapsed'],
                            "roc time": param_result[1].result['time_elapsed'],
                            "roc value": param_result[1].result['fun_return'],
                        }
                    # for param_result in list(zip(data_fit, data_roc))
                    for param_result in list(zip(data_fit, data_roc, range(len(data_fit))))

                },
                orient='index'
            )
    

    if len(df):
        df.index = df.index.set_names(['path', 'model', 'run_id'])

    df_t = df.reset_index()

    def insert_run_id(g):
        print(type(g))
        # num_runs = len(g)
        # g['run_id'] = range(len(num_runs))
        return g
    groups = df_t.groupby(['path', 'model'])
    for g in groups:
        print(type(g[1]))


    return df



def get_df_class_id_model(df):
    if not len(df):
        return df
    def split_path(row):
        path = row.name[0].strip()
        row['id'] = path
        row['class'] = path
        if(len(path) > 3):
            row['id'] = path.split('/')[-2] 
            row['class'] = path.split('/')[-3]
        return row
    df = df.apply(split_path, axis=1)
    df = df.reset_index()
    df = df.drop(columns='path')
    df = df.set_index(['class', 'id', 'model', 'run id'])
    return df


def get_df_aggregate(df, index, values, df_slice=None, aggfunc=None):
    if df_slice:
        df = df.loc[df_slice]
    all_index = ['class', 'id', 'run id']
    columns = 'model'
    if not aggfunc:
        aggfunc = ['mean']
    df = df.reset_index()
    df = df.pivot_table(index=index, values=values, aggfunc=aggfunc, columns=columns)
    for i in all_index:
        if(not i in index):
            df[i] = 'all'
    df = df.reset_index()
    df = df.set_index(['class', 'id', 'run id'])
    return df

def get_df_from_path(path):
    df = read_df_from_json(path)
    df = get_df_class_id_model(df)
    return df

