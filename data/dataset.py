import torch
from torch.utils.data import Dataset, Sampler
import h5py
import numpy as np

from torch.utils.data import Sampler, BatchSampler, DataLoader
import torch
import torchdata.datapipes as dp


def string_mapper(string, substring="GENIE"):
    if substring in string:
        return 1
    else:
        return 0
def feature_mapper(feature):
    if type(feature) == str:
        return 0
    else:
        return feature

def row_processor(row):
    label, features = row
    label = string_mapper(label)
    features = [feature_mapper(x) for x in features]
    features = features[1:]  # remove index from csv
    return {
        "label": np.array(label, np.int32),
        "features": np.array(features, np.float32),
    }


def build_datapipes(root_dir=".", mode='train'):
    mask = f'*{mode}*.csv'
    datapipe = dp.iter.FileLister(root_dir, masks=mask)
    datapipe = dp.iter.FileOpener(datapipe, mode="rt")
    # Shuffle filenames
    datapipe = datapipe.shuffle()

    # Parse filenames
    datapipe = datapipe.parse_csv(delimiter=",", skip_lines=1, return_path=True)

    # if buffer size is bigger than the lenght of the shard it shuffles across shards
    datapipe = datapipe.shuffle(buffer_size=10e5)

    datapipe = datapipe.map(row_processor)
    return datapipe


class RandomBatchSampler(Sampler):
    """Sampling class to create random sequential batches from a given dataset
    E.g. if data is [1,2,3,4] with bs=2. Then first batch, [[1,2], [3,4]] then shuffle batches -> [[3,4],[1,2]]
    This is useful for cases when you are interested in 'weak shuffling'
    :param dataset: dataset you want to batch
    :type dataset: torch.utils.data.Dataset
    :param batch_size: batch size
    :type batch_size: int
    :returns: generator object of shuffled batch indices
    """

    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.dataset_length = len(dataset)
        self.n_batches = self.dataset_length / self.batch_size
        self.batch_ids = torch.randperm(int(self.n_batches))

    def __len__(self):
        return self.batch_size

    def __iter__(self):
        for id in self.batch_ids:
            idx = torch.arange(id * self.batch_size, (id + 1) * self.batch_size)
            for index in idx:
                yield int(index)
        if int(self.n_batches) < self.n_batches:
            idx = torch.arange(
                int(self.n_batches) * self.batch_size, self.dataset_length
            )
            for index in idx:
                yield int(index)


def fast_loader(dataset, batch_size=32, drop_last=False, transforms=None):
    """Implements fast loading by taking advantage of .h5 dataset
    The .h5 dataset has a speed bottleneck that scales (roughly) linearly with the number
    of calls made to it. This is because when queries are made to it, a search is made to find
    the data item at that index. However, once the start index has been found, taking the next items
    does not require any more significant computation. So indexing data[start_index: start_index+batch_size]
    is almost the same as just data[start_index]. The fast loading scheme takes advantage of this. However,
    because the goal is NOT to load the entirety of the data in memory at once, weak shuffling is used instead of
    strong shuffling.
    :param dataset: a dataset that loads data from .h5 files
    :type dataset: torch.utils.data.Dataset
    :param batch_size: size of data to batch
    :type batch_size: int
    :param drop_last: flag to indicate if last batch will be dropped (if size < batch_size)
    :type drop_last: bool
    :returns: dataloading that queries from data using shuffled batches
    :rtype: torch.utils.data.DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=None,  # must be disabled when using samplers
        sampler=BatchSampler(
            RandomBatchSampler(dataset, batch_size),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
    )


class PairedDataset(Dataset):
    def __init__(self, x, z, shuffle=False, decouple=False):
        super(PairedDataset, self).__init__()

        self.length = min([len(x), len(z)])
        self.x = torch.tensor(x[: self.length])
        self.z = torch.tensor(z[: self.length])

        if decouple:
            perm = torch.randperm(self.x.size(0))
            self.x = self.x[perm]

        if shuffle:
            perm = torch.randperm(self.x.size(0))
            self.x = self.x[perm]
            self.z = self.z[perm]

    def __len__(self):
        return self.length


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
