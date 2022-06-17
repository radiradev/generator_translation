import pandas as pd
import h5py 

#Uses data_create_old.py to create h5py files first ---> quickfix solution 

DIR = '/eos/user/r/rradev/generator_reweigthing/'
GENIE_NAME = 'argon_GENIEv2.h5'
NEUT_NAME = 'argon_NEUT.h5'
N_CHUNKS = 20
sample_names = [GENIE_NAME, NEUT_NAME]
col_names = ["isNu", "isNue", "isNumu", "isNutau", "cc", "Enu_true", "ELep", "CosLep", "Q2", "W", "x", "y", "nP", "nN", "nipip", "nipim", "nipi0", "niem", "eP", "eN", "ePip", "ePim", "ePi0"]

lengths = {'train_data': 16200000, 'test_data': 1800000}

sample = h5py.File(DIR + NEUT_NAME)
for sample_name in sample_names:
    sample = h5py.File(DIR + sample_name)
    prev_chunk_idx = 0
    for split in ['test_data', 'train_data']:
        chunk_size = lengths[split]/N_CHUNKS
        for chunk in range(N_CHUNKS):
            chunk_index = int((chunk+1)*chunk_size)
            data = sample[split][:lengths[split]]
            data = data[prev_chunk_idx:chunk_index]
            df = pd.DataFrame(data, columns=col_names)
            print(str(chunk) + sample_name)
            df.to_csv(DIR + '/from_h5/' + split + str(chunk) + '_' + sample_name[:-3] + '.csv')

            prev_chunk_idx = chunk_index

            


    