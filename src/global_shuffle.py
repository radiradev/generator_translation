import dask.dataframe as dd
import numpy as np
import pandas as pd
import glob
import ray


root_dir = '/eos/user/r/rradev/generator_reweigthing'
print('reading paths')

paths = glob.glob(root_dir + '/*.parquet')
ray_df = ray.data.read_parquet(paths)
ray_df = ray_df.repartition(num_blocks=100, shuffle=True)

# Return a dask dataframe
dask_df =  ray_df.to_dask()
dask_df.to_csv(root_dir + '/global_shuffle/' + '*_train_' + '.csv')
