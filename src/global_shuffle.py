import dask.dataframe as dd
import numpy as np
import pandas as pd
import dask.array as da
from sklearn.utils import shuffle

def set_random_indices(df):
    indices = np.arange(len(df))
    chunks = tuple(df.map_partitions(len).compute())
    size = sum(chunks)
    rand_indices = da.from_array(np.random.choice(indices, size=len(indices), replace=False), chunks=chunks)
    df['rand_index'] = rand_indices
    return df

def shuffle_data(generator_name, root_dir='/eos/home-r/rradev/generator_reweigthing/from_h5/'):
    mask = f'*train_data*{generator_name}*.csv'
    df = dd.read_csv(root_dir + mask)
    df = set_random_indices(df)
    df = df.drop(columns=['Unnamed: 0', 'rand_index'])
    return df

generator_names = ['GENIE, NEUT']
root_dir = '/eos/home-r/rradev/generator_reweigthing/from_h5/'

for generator_name in generator_names:
    df = shuffle(generator_name)
    df.to_csv(root_dir + '/global_shuffle/' + 'train_'+ generator_name + '_*.csv') 
