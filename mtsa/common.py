import time
from functools import partial
from functools import reduce
from mtsa.experiments import *

def elapsed_time(fun, *args) -> Result:
    start = time.time()
    fun_return = fun(*args)
    end = time.time()
    time_elapsed = end - start
    result = Result(time_elapsed, fun_return) 
    return result


def multiple_runs(runs, fun, *args):
    list_run_ids = range(runs)
    multiple_results = map(
        lambda run_id:
        {
            'run_id': run_id,
            "run_details": elapsed_time(fun, *args)
        },
        list_run_ids
        )

    return list(multiple_results)


import json
import functools as ft
def print(element):
    #TODO this is ugly but leting keras to be serialized causes circular reference at json.dumps 
    should_dump = lambda o: any(map(ft.partial(isinstance, o), [t for t in [int, float, str, ParamResult, Result, Params, ModelParam]]))
    def default(o):
        if should_dump(o):
            if hasattr(o, "__dict__"):
                return o.__dict__
            else:
                return str(o)
        else:
            return ""
    json_data = json.dumps(element, indent=4, default=default)
    return json