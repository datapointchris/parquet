import itertools
import multiprocessing as mp
import timeit
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

NUM_RECORDS = range(100, 2500, 100)
RECORD_SIZES = range(100, 2500, 100)


def compare_methods(record_size, num_records, iterations=2):
    def pyarrow():
        return pa.Table.from_pydict(sample_data)

    def pandas():
        return pa.Table.from_pandas(pd.DataFrame(sample_data))

    sample_data = {str(i): list(range(i, i + num_records)) for i in range(record_size)}

    pyarrow_time = timeit.timeit(pyarrow, number=iterations)
    pandas_time = timeit.timeit(pandas, number=iterations)

    fastest = "Pyarrow" if pyarrow_time < pandas_time else "Pandas"
    percentage_time_difference = (abs(pyarrow_time - pandas_time) / max(pyarrow_time, pandas_time)) * 100

    return {
        'record_size': record_size,
        'num_records': num_records,
        'pyarrow_time': round(pyarrow_time, 4),
        'pandas_time': round(pandas_time, 4),
        'fastest': fastest,
        'percentage_difference': round(percentage_time_difference, 2),
    }


def process_record_pairs(record):
    record_size, num_records = record
    # start_time = timeit.default_timer()
    d = compare_methods(record_size=record_size, num_records=num_records)
    # print(f'{record_size} record size, {num_records} records: {timeit.default_timer() - start_time:.2f} seconds')
    return d


def plot_results(df):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Pyarrow Vs Pandas', fontsize=16)

    ax[0, 0].set_title('Record Size Vs Time')
    ax[0, 0].set_xlabel('Record Size (#)')
    ax[0, 0].set_ylabel('Time (s)')

    # Fit a polynomial to the pyarrow data
    z_pyarrow = np.polyfit(df['record_size'], df['pyarrow_time'], 1)
    p_pyarrow = np.poly1d(z_pyarrow)
    ax[0, 0].plot(df['record_size'], p_pyarrow(df['record_size']), 'g--')
    ax[0, 0].plot(df['record_size'], df['pyarrow_time'], 'o', label='pyarrow')

    # Fit a polynomial to the pandas data
    z_pandas = np.polyfit(df['record_size'], df['pandas_time'], 1)
    p_pandas = np.poly1d(z_pandas)
    ax[0, 0].plot(df['record_size'], p_pandas(df['record_size']), 'r--')
    ax[0, 0].plot(df['record_size'], df['pandas_time'], 'o', label='pandas')

    ax[0, 0].legend()

    ax[0, 1].set_title('Record Size Vs Percentage Difference')
    ax[0, 1].set_xlabel('Record Size (#)')
    ax[0, 1].set_ylabel('Percentage Difference (%)')
    z_percentage = np.polyfit(df['record_size'], df['percentage_difference'], 1)
    p_percentage = np.poly1d(z_percentage)
    ax[0, 1].plot(df['record_size'], p_percentage(df['record_size']), 'r--')
    ax[0, 1].plot(df['record_size'], df['percentage_difference'], 'o')

    ax[1, 0].set_title('Number of Records Vs Time')
    ax[1, 0].set_xlabel('Number of Records (#)')
    ax[1, 0].set_ylabel('Time (s)')

    # Fit a polynomial to the pyarrow data
    z_pyarrow = np.polyfit(df['num_records'], df['pyarrow_time'], 1)
    p_pyarrow = np.poly1d(z_pyarrow)
    ax[1, 0].plot(df['num_records'], p_pyarrow(df['num_records']), 'g--')
    ax[1, 0].plot(df['num_records'], df['pyarrow_time'], 'o', label='pyarrow')

    # Fit a polynomial to the pandas data
    z_pandas = np.polyfit(df['num_records'], df['pandas_time'], 1)
    p_pandas = np.poly1d(z_pandas)
    ax[1, 0].plot(df['num_records'], p_pandas(df['num_records']), 'r--')
    ax[1, 0].plot(df['num_records'], df['pandas_time'], 'o', label='pandas')

    ax[1, 0].legend()

    ax[1, 1].set_title('Number of Records Vs Percentage Difference')
    ax[1, 1].set_xlabel('Number of Records (#)')
    ax[1, 1].set_ylabel('Percentage Difference (%)')
    z_percentage = np.polyfit(df['num_records'], df['percentage_difference'], 1)
    p_percentage = np.poly1d(z_percentage)
    ax[1, 1].plot(df['num_records'], p_percentage(df['num_records']), 'r--')
    ax[1, 1].plot(df['num_records'], df['percentage_difference'], 'o')

    plt.tight_layout(pad=3.0)
    fig.tight_layout(pad=3.0)

    plt.show()


if __name__ == '__main__':
    start_time = timeit.default_timer()
    print(f'Number of record pairs: {len(list(itertools.product(RECORD_SIZES, NUM_RECORDS)))}')
    with mp.Pool() as pool:
        results = pool.map(process_record_pairs, itertools.product(RECORD_SIZES, NUM_RECORDS))
    print(f'Total time: {timeit.default_timer() - start_time:.2f} seconds')

    df = pd.DataFrame(results)
    filename = f'{datetime.now().isoformat()}_{len(NUM_RECORDS)}x{len(RECORD_SIZES)}.csv'

    with open(filename, 'w') as f:
        df.to_csv(f, index=False)
    print(f'Results saved to {filename}\n')

    plot_results(df)
