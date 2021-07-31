import multiprocessing
import traceback
from tqdm import tqdm

import pandas as pd
import dask
import dask.bag as db
from dask.diagnostics import ProgressBar

# helper functions
# ===============
def interpret_num_workers(num_workers):
    if num_workers < 1:
        num_workers = multiprocessing.cpu_count()
    return num_workers

def get_catch_all_warpper(func):
    def wrapper(*args, **kwargs):
        result = None
        err = None
        stack_trace = None
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            err = e
            stack_trace = traceback.format_exc()
        return {
            'result': result,
            'error': err,
            'stack_trace': stack_trace
        }
    return wrapper

# high level API
# ==============
def parallel_map(
        func, sequence,
        num_workers=-1,
        scheduler="processes",
        include_input=False,
        preserve_index=True
        ):
    # process input
    num_workers = interpret_num_workers(num_workers)
    input_sequence = list(sequence)
    safe_func = get_catch_all_warpper(func)

    # map
    output_sequence = None
    if num_workers == 1:
        output_sequence = list()
        for x in tqdm(input_sequence):
            output_sequence.append(safe_func(x))
    else:
        input_sequence_b = db.from_sequence(input_sequence)
        output_sequence_b = input_sequence_b.map(safe_func)
        with dask.config.set({
            'scheduler': scheduler,
            'multiprocessing.context': 'fork',
            'num_workers': num_workers
            }):
            with ProgressBar():
                output_sequence = output_sequence_b.compute()
    
    # consolidate result
    index = None
    if isinstance(sequence, pd.Series) and preserve_index:
        index = sequence.index
    result_df = pd.DataFrame(output_sequence, 
        columns=['result', 'error', 'stack_trace'],
        index=index)
    if include_input:
        result_df['input'] = input_sequence
    return result_df
