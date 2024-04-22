import itertools
import multiprocessing as mp
import sys
import timeit
from datetime import datetime
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

NUM_RECORDS = range(100, 1010, 10)
RECORD_SIZES = range(100, 1010, 10)


def plot_results(df, show=False):
    fig, ax = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Pyarrow Vs Pandas', fontsize=16)

    ax[0, 0].set_title('Record Size Vs Time')
    ax[0, 0].set_xlabel('Record Size (#)')
    ax[0, 0].set_ylabel('Time (s)')
    z_pyarrow = np.polyfit(df['record_size'], df['pyarrow_time'], 1)
    z_pandas = np.polyfit(df['record_size'], df['pandas_time'], 1)
    p_pyarrow = np.poly1d(z_pyarrow)
    p_pandas = np.poly1d(z_pandas)
    ax[0, 0].plot(df['record_size'], df['pandas_time'], 'o', label='pandas')
    ax[0, 0].plot(df['record_size'], df['pyarrow_time'], 'o', label='pyarrow')
    ax[0, 0].plot(df['record_size'], p_pandas(df['record_size']), 'r--')
    ax[0, 0].plot(df['record_size'], p_pyarrow(df['record_size']), 'g--')
    ax[0, 0].legend()

    ax[0, 1].set_title('Record Size Vs Percentage Difference')
    ax[0, 1].set_xlabel('Record Size (#)')
    ax[0, 1].set_ylabel('Percentage Difference (%)')
    z_percentage = np.polyfit(df['record_size'], df['percentage_time_difference'], 1)
    p_percentage = np.poly1d(z_percentage)
    ax[0, 1].plot(df['record_size'], df['percentage_time_difference'], 'o')
    ax[0, 1].plot(df['record_size'], p_percentage(df['record_size']), 'r--')

    ax[1, 0].set_title('Number of Records Vs Time')
    ax[1, 0].set_xlabel('Number of Records (#)')
    ax[1, 0].set_ylabel('Time (s)')
    z_pyarrow = np.polyfit(df['num_records'], df['pyarrow_time'], 1)
    z_pandas = np.polyfit(df['num_records'], df['pandas_time'], 1)
    p_pyarrow = np.poly1d(z_pyarrow)
    p_pandas = np.poly1d(z_pandas)
    ax[1, 0].plot(df['num_records'], df['pandas_time'], 'o', label='pandas')
    ax[1, 0].plot(df['num_records'], df['pyarrow_time'], 'o', label='pyarrow')
    ax[1, 0].plot(df['num_records'], p_pandas(df['num_records']), 'r--')
    ax[1, 0].plot(df['num_records'], p_pyarrow(df['num_records']), 'g--')
    ax[1, 0].legend()

    ax[1, 1].set_title('Number of Records Vs Percentage Difference')
    ax[1, 1].set_xlabel('Number of Records (#)')
    ax[1, 1].set_ylabel('Percentage Difference (%)')
    z_percentage = np.polyfit(df['num_records'], df['percentage_time_difference'], 1)
    p_percentage = np.poly1d(z_percentage)
    ax[1, 1].plot(df['num_records'], df['percentage_time_difference'], 'o')
    ax[1, 1].plot(df['num_records'], p_percentage(df['num_records']), 'r--')

    ax[2, 0].set_title('Pyarrow Table Size Vs Total Items (Records * Columns)')
    ax[2, 0].set_xlabel('Total Items')
    ax[2, 0].ticklabel_format(style='plain', axis='x')  # disable scientific notation
    ax[2, 0].set_ylabel('Table Size (MB)')
    ax[2, 0].plot(df['total_records'], df['pyarrow_table_size'], 'o', label='pyarrow')
    y2_y1 = df['pyarrow_table_size'].iloc[-1] - df['pyarrow_table_size'].iloc[0]
    x2_x1 = df['total_records'].iloc[-1] - df['total_records'].iloc[0]
    slope = y2_y1 / x2_x1
    ax[2, 0].annotate('Table Size (MB / 1000 items): ' + str(slope * 1000), xy=(0.05, 0.95), xycoords='axes fraction')

    ax[2, 1].set_title('Pyarrow Table Size Vs Number of Records')
    ax[2, 1].set_xlabel('Number of Records (#)')
    ax[2, 1].set_ylabel('Time (s)')
    ax[2, 1].scatter(
        df['num_records'],
        df['pyarrow_time'],
        s=df['pyarrow_table_size'],
        c=df['record_size'],
        cmap=matplotlib.colormaps['binary'],
    )
    ax[2, 1].annotate('Size: Table Size MB', xy=(0.05, 0.95), xycoords='axes fraction')
    ax[2, 1].annotate('Color: Record Size', xy=(0.05, 0.90), xycoords='axes fraction')

    plt.tight_layout(pad=3.0)
    fig.tight_layout(pad=3.0)
    if show:
        plt.show()
    return fig


def compare_methods(record_size, num_records, iterations=1):
    pyarrow_size_mb = 0
    pandas_size_mb = 0

    def pyarrow():
        table = pa.Table.from_pydict(sample_data)
        nonlocal pyarrow_size_mb
        pyarrow_size_mb = sys.getsizeof(table) / 1024 / 1024

    def pandas():
        table = pa.Table.from_pandas(pd.DataFrame(sample_data))
        nonlocal pandas_size_mb
        pandas_size_mb = sys.getsizeof(table) / 1024 / 1024

    sample_data = {str(i): list(range(i, i + num_records)) for i in range(record_size)}

    pyarrow_time = timeit.timeit(pyarrow, number=iterations)
    pandas_time = timeit.timeit(pandas, number=iterations)
    percentage_time_difference = (abs(pyarrow_time - pandas_time) / min(pyarrow_time, pandas_time)) * 100
    percentage_size_difference = (abs(pyarrow_size_mb - pandas_size_mb) / max(pyarrow_size_mb, pandas_size_mb)) * 100

    return {
        'record_size': record_size,
        'num_records': num_records,
        'total_records': record_size * num_records,
        'pyarrow_time': round(pyarrow_time, 4),
        'pandas_time': round(pandas_time, 4),
        'percentage_time_difference': round(percentage_time_difference, 2),
        'pyarrow_table_size': pyarrow_size_mb,
        'pandas_table_size': pandas_size_mb,
        'percentage_size_difference': round(percentage_size_difference, 2),
        'fastest': 'Pyarrow' if pyarrow_time < pandas_time else 'Pandas',
    }


def format_time(seconds: float):
    minutes = int(seconds // 60)
    hours = int(minutes // 60)
    if hours > 0:
        return f'{hours} hours {minutes % 60} minutes {int(seconds % 60)} seconds'
    elif minutes > 0:
        return f'{minutes} minutes {int(seconds % 60)} seconds'
    return f'{seconds:.2f} seconds'


def process_record_pairs(record, counter, total_records, start_time):
    record_size, num_records = record
    if counter.value % 100 == 0 and counter.value != 0:
        print(f'Processing record {counter.value} of {total_records}')
        print(f'Elapsed time: {format_time(timeit.default_timer() - start_time)}')
    d = compare_methods(record_size=record_size, num_records=num_records)
    counter.value += 1
    return d


if __name__ == '__main__':
    start_time = timeit.default_timer()
    total_records = len(list(itertools.product(RECORD_SIZES, NUM_RECORDS)))
    print(f'Number of record pairs: {total_records}')

    counter = mp.Manager().Value('i', 0)  # shared counter between processes
    partial_process_record_pairs = partial(
        process_record_pairs,
        counter=counter,
        total_records=total_records,
        start_time=start_time,
    )

    with mp.Pool() as pool:
        results = pool.map(partial_process_record_pairs, itertools.product(RECORD_SIZES, NUM_RECORDS))
    print('Finished Processing')
    print(f'Total time: {format_time(timeit.default_timer() - start_time)}')

    df = pd.DataFrame(results)
    filename = f'{datetime.now().isoformat(timespec="seconds")}_{len(NUM_RECORDS)}x{len(RECORD_SIZES)}'

    with open(filename + '.csv', 'w') as f:
        df.to_csv(f, index=False)
    print(f'Results saved to {filename}')

    fig = plot_results(df, show=False)
    fig.savefig(f'{filename}.png')
    print(f'Plot saved to {filename}.png')
