import time
from functools import partial
from functools import reduce
from scripts.experiments import Result

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

