import pandas as pd
import glob
# from scripts.experiments import *
import json

COLUMN_PATH = 'path'
COLUMN_MODEL = 'model'
COLUMN_METRIC_NAME = 'name'
COLUMN_METRIC_VALUE = 'value'
COLUNM_RUNID = 'run_id'

def read_data(path):
    file_paths = glob.glob(f'{path}*.text*')
    dfs = []
    for file_path in file_paths:
        with open(file_path) as f:
            lines = [json.loads(line.replace("\'", "\"")) for line in f]
            df = pd.DataFrame(
                [
                    {
                        COLUMN_PATH: index[COLUMN_PATH],
                        COLUMN_MODEL: index[COLUMN_MODEL],
                        COLUMN_METRIC_NAME: row[COLUMN_METRIC_NAME],
                        COLUMN_METRIC_VALUE: row[COLUMN_METRIC_VALUE],
                    }
                    for index in lines for row in index['metrics']
                ]
            )
            dfs.append(df)
        
    df = pd.concat(dfs)
    def insert_run_id(g):
        num_runs = len(g)
        g[COLUNM_RUNID] = list(range(num_runs))
        return g
    df = df.groupby([COLUMN_PATH, COLUMN_MODEL, COLUMN_METRIC_NAME], group_keys=False).apply(insert_run_id)
    df = df.pivot(index=[COLUMN_PATH, COLUMN_MODEL, COLUNM_RUNID], columns=COLUMN_METRIC_NAME, values=COLUMN_METRIC_VALUE).reset_index()
    df = df.set_index([COLUMN_PATH, COLUMN_MODEL, COLUNM_RUNID])
    return df

def get_df_class_id_model(df):
    if not len(df):
        return df

    def split_path(row):
        path = row.name[0].strip()
        row['id'] = path
        row['class'] = path
        if (len(path) > 3):
            row['id'] = path.split('/')[-2]
            row['class'] = path.split('/')[-3]
        return row
    df = df.apply(split_path, axis=1)
    df = df.reset_index()
    df = df.drop(columns='path')
    df = df.set_index(['class', 'id', 'model', COLUNM_RUNID])
    return df

def get_df_aggregate(df, index, values, df_slice=None, aggfunc=None):
    if df_slice:
        df = df.loc[df_slice]
    all_index = ['class', 'id', COLUNM_RUNID]
    columns = 'model'
    if not aggfunc:
        aggfunc = ['mean', 'std']
    df = df.reset_index()
    df = df.pivot_table(index=index, values=values,
                        aggfunc=aggfunc, columns=columns)
    for i in all_index:
        if (not i in index):
            df[i] = 'all'
    df = df.reset_index()
    df = df.set_index(['class', 'id', COLUNM_RUNID])
    return df


def get_df_from_path(path):
    df = read_data(path)
    # df = normalize(df)
    df = get_df_class_id_model(df)
    return df
