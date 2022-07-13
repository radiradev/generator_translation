from distutils import filelist
import torch
from torch.utils.data import Dataset, Sampler
import h5py
import numpy as np

from torch.utils.data import Sampler, BatchSampler, DataLoader
import torch
import torchdata.datapipes as dp


def build_datapipes(root_dir=".", mode="train", buffer_size=10e4):
    mask = f"*{mode}*.csv"
    filelister = dp.iter.FileLister(root_dir, masks=mask)
    datapipe = dp.iter.FileOpener(filelister, mode="rt")

    # Parse filenames
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1, return_path=False)

    # if buffer size is bigger than the lenght of the shard it shuffles across shards
    datapipe = datapipe.shuffle(buffer_size=buffer_size)

    datapipe = datapipe.map(row_processor)
    return datapipe, filelister


def row_processor(row):
    # Ignore index column and select last col as labels
    features, label = row[:-1], row[-1]
    features = features_preprocess(features)
    return {
        "label": np.array(label, np.int32),
        "features": np.array(features, np.float32),
    }


def features_preprocess(features):
    features = features[1:] 
    # nan values fi
    W = features[9]
    if W == "":
        features[9] = np.random.random() * 10
    return features


class NuDataset(Dataset):
    def __init__(self, file_names, test=False):
        print("INITIALISING DATASET")
        self._lengths = []
        self._overallLength = 0

        if test:
            self._key = "test_data"
        else:
            self._key = "train_data"

        self._file_names = file_names

        for i, filename in enumerate(file_names):
            print("APPENDING FILE {0}".format(i))
            with h5py.File(filename, "r") as f:
                self._lengths.append(len(f[self._key]))
                self._overallLength += self._lengths[-1]

        self._fileEnds = np.cumsum(self._lengths)
        print("ENDS INITIALISATION")

    def __len__(self):
        return self._overallLength

    def __getitem__(self, i):
        fileNumber = np.digitize(i, self._fileEnds)
        this_i = i if fileNumber == 0 else i - self._fileEnds[fileNumber - 1]

        with h5py.File(self._file_names[fileNumber], "r") as f:
            features = f[self._key][this_i]

        # W is NaN in 0.01% of the NUWRO file. For now replace with uniformly distributed number in the [0, 10] GeV/c^2 range
        if np.isnan(features[9]):
            features[9] = np.random.random() * 10

        if sum(np.isnan(features)) > 0:
            print("DATASET FOUND NAN")
            print(features)
            print("FILE {0} i {1}".format(fileNumber, this_i))
            exit(-1)

        return {"features": features, "label": fileNumber}
