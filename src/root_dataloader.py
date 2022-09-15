import itertools
import uproot
import awkward as ak
import numpy as np
from glob import glob
from torch.utils.data import Dataset

from utils.funcs import get_constants


class NumPyDataset(Dataset):
    """
    Loads converted ROOT files (numpy) from two generators into memory
    """

    def __init__(self,
                 root_dir,
                 generator_a='GENIEv2',
                 generator_b='NUWRO',
                 number_of_files=1,
                 shuffle=True):
        super().__init__()
        self.root_dir = root_dir
        self.generator_a = generator_a
        self.generator__b = generator_b
        self.shuffle = shuffle
        self.number_of_files = number_of_files
        self.dataset_a = self.load_generator(generator_a)
        self.dataset_b = self.load_generator(generator_b)
        self.data = self.load_data(self.dataset_a, self.dataset_b)

    def __getitem__(self, index):
        features, label = self.data[index][:-1], self.data[index][-1]
        # Replace nan features
        if np.isnan(features[9]):
            features[9] = np.random.random() * 10

        return {'features': features, 'label': label}

    def __len__(self):
        return len(self.data)

    def draw_index(self, max_index=50):                    
        return np.random.randint(low=0,high=max_index)

    def stack(self):
        pass


    def load_generator(self, generator_name):
        index = self.draw_index()
        data = np.load(self.root_dir + generator_name + str(index) + '.npy')
        if self.number_of_files > 1:
            for _ in range(self.number_of_files):
                index += 1 
                print(f'{generator_name}{index}')
                new_data = np.load(self.root_dir + generator_name + str(index) + '.npy')
                data = np.append(data, new_data, axis=0)

        labels = self.create_labels(data, generator_name)
        return np.hstack([data, labels])

    def load_data(self, dataset_a, dataset_b):
        data = np.vstack([dataset_a, dataset_b])
        if self.shuffle:
            np.random.shuffle(data)
        return data

    def create_labels(self, data, filename):
        if self.generator_a in filename:
            labels = np.zeros(len(data))
        else:
            labels = np.ones(len(data))
        return np.expand_dims(labels, axis=1)


class ROOTDataset(Dataset):
    """
    Loads ROOT files of PDG types 12, -12, 14, -14 from two generators into memory.

    """

    def __init__(self,
                 data_dir,
                 generator_a='GENIEv2',
                 generator_b='NUWRO',
                 shuffle=True,
                 preload_data=True,
                 sqrt_counts=False,
                 test_data=False,
                 gibuu_dir='/afs/cern.ch/work/r/rradev/gibuu_files/',
                 number_of_indices=1):
        super().__init__()
        self.data_dir = data_dir
        self.generator_a = generator_a
        self.generator_b = generator_b
        self.weights_sum_a = 0
        self.weights_sum_b = 0
        self.gibuu_dir = gibuu_dir
        self.shuffle = shuffle
        self.sqrt_counts = sqrt_counts
        self.test_data = test_data
        self.number_of_indices = number_of_indices
        if preload_data:
            self.dataset_a = self.load_generator(generator_a)
            self.dataset_b = self.load_generator(generator_b)
            self.data = self.load_data(self.dataset_a, self.dataset_b)

    def __getitem__(self, index):

        features, label, weights = self.data[index][:-2], self.data[index][
            -2], self.data[index][-1]

        # Replace nan features
        if np.isnan(features[9]):
            features[9] = np.random.random() * 10

        return {'features': features, 'label': label, 'weights': weights}

    def __len__(self):
        return len(self.data)

    def load_data(self, dataset_a, dataset_b):
        data = np.vstack([dataset_a, dataset_b])
        if self.shuffle:
            np.random.shuffle(data)
        return data

    def load_generator(self, generator_name):
        """
        Loads generator file
        """
        data_dir = self.data_dir
        filenames = self.get_filenames(generator_name, data_dir)

        # Merged GiBUU files are in a different directory
        if generator_name == 'GiBUU':
            data_dir = self.gibuu_dir

        # Debug filename matching
        assert len(
            filenames
        ) != 0, f"Matched files are zero. Trying to retrive files {filenames}"

        # The dataset has to include all pdg types (should not be a big issue for GiBUU)
        len_pdg_types = 4
        while len(filenames) % len_pdg_types == 0:
            filenames = self.get_filenames(generator_name, data_dir)

        data, weights = self._rootfile_to_array(filenames[0])

        for filename in filenames[1:]:
            new_data, new_weights = self._rootfile_to_array(filename)
            data = np.append(data, new_data, axis=0)
            weights = np.append(weights, new_weights, axis=0)
            if generator_name == 'GiBUU':
                self.weights_sum_b = np.sum(weights)
            else:
                self.weights_sum_a = np.sum(weights)

        if generator_name == 'GiBUU':
            weights = self.normalize_weights(weights, self.weights_sum_a,
                                             self.weights_sum_b)
        # Create Labels
        labels = self.create_labels(data, filename)
        return np.hstack([data, labels, weights])

    def normalize_weights(self, weights, weights_sum_a, weights_sum_b):
        assert weights_sum_a != 0 or weights_sum_b != 0, 'Weight sums are zero'
        ratio = weights_sum_b / weights_sum_a
        print(f'Weight ratio is {ratio} diving by this to normalize weights')
        return weights / ratio

    def create_labels(self, data, filename):
        if self.generator_a in filename:
            labels = np.zeros(len(data))
        else:
            labels = np.ones(len(data))
        return np.expand_dims(labels, axis=1)

    def draw_file_indices(self, max_file_index=50, number_of_indices=1):
        """
        Draws random indices to create a dataset
        """
        indices = np.random.randint(low=0,
                                    high=max_file_index,
                                    size=number_of_indices)
        return indices

    def get_filenames(self, generator_name, data_dir, max_file_index=50):
        # String formatting to get the filename in the correct way
        if generator_name == 'GiBUU':
            max_file_index = 4

        indices = self.draw_file_indices(max_file_index,
                                         self.number_of_indices)

        # filenames = []
        # for index in indices:

        #     filenames.append()
        filenames = [
            glob(data_dir + self.format_filename(generator_name, index))
            for index in indices
        ]
        flatttened_filenames = list(itertools.chain(*filenames))

        return flatttened_filenames

    def format_filename(self, generator_name, index):
        """
        Creates a ROOT filename wildcard with the passed index
        """
        formatted_index = str(index).zfill(3)
        filename_pattern = f"*{generator_name}*_1M_{formatted_index}_NUISFLAT.root"
        return filename_pattern

    def _rootfile_to_array(self, filename):
        variables_in, variables_out, m = get_constants()

        with uproot.open(filename + ":FlatTree_VARS") as tree:
            print("Reading {0}".format(filename))
            treeArr = tree.arrays(variables_in)

            Lepmask = ((treeArr["pdg"] == 11) + (treeArr["pdg"] == -11) +
                       (treeArr["pdg"] == 13) + (treeArr["pdg"] == -13) +
                       (treeArr["pdg"] == 15) + (treeArr["pdg"] == -15))
            Numask = ((treeArr["pdg"] == 12) + (treeArr["pdg"] == -12) +
                      (treeArr["pdg"] == 14) + (treeArr["pdg"] == -14) +
                      (treeArr["pdg"] == 16) + (treeArr["pdg"] == -16))
            Pmask = treeArr["pdg"] == 2212
            Nmask = treeArr["pdg"] == 2112
            Pipmask = treeArr["pdg"] == 211
            Pimmask = treeArr["pdg"] == -211
            Pi0mask = treeArr["pdg"] == 111
            Kpmask = treeArr["pdg"] == 321
            Kmmask = treeArr["pdg"] == -321
            K0mask = ((treeArr["pdg"] == 311) + (treeArr["pdg"] == -311) +
                      (treeArr["pdg"] == 130) + (treeArr["pdg"] == 310))
            EMmask = treeArr["pdg"] == 22

            othermask = (Numask + Lepmask + Pmask + Nmask + Pipmask + Pimmask +
                         Pi0mask + Kpmask + Kmmask + K0mask + EMmask) == False

            # Count particles based on PDG type
            treeArr["nP"] = ak.count_nonzero(Pmask, axis=1)
            treeArr["nN"] = ak.count_nonzero(Nmask, axis=1)
            treeArr["nipip"] = ak.count_nonzero(Pipmask, axis=1)
            treeArr["nipim"] = ak.count_nonzero(Pimmask, axis=1)
            treeArr["nipi0"] = ak.count_nonzero(Pi0mask, axis=1)
            treeArr["nikp"] = ak.count_nonzero(Kpmask, axis=1)
            treeArr["nikm"] = ak.count_nonzero(Kmmask, axis=1)
            treeArr["nik0"] = ak.count_nonzero(K0mask, axis=1)
            treeArr["niem"] = ak.count_nonzero(EMmask, axis=1)

            # Energy per particle type
            treeArr["eP"] = ak.sum(treeArr["E"][Pmask],
                                   axis=1) - treeArr["nP"] * m["P"]
            treeArr["eN"] = ak.sum(treeArr["E"][Nmask],
                                   axis=1) - treeArr["nN"] * m["N"]
            treeArr["ePip"] = (ak.sum(treeArr["E"][Pipmask], axis=1) -
                               treeArr["nipip"] * m["piC"])
            treeArr["ePim"] = (ak.sum(treeArr["E"][Pimmask], axis=1) -
                               treeArr["nipim"] * m["piC"])
            treeArr["ePi0"] = (ak.sum(treeArr["E"][Pi0mask], axis=1) -
                               treeArr["nipi0"] * m["pi0"])

            treeArr["eOther"] = ak.sum(
                treeArr["E"][othermask] -
                (treeArr["E"][othermask]**2 - treeArr["px"][othermask]**2 -
                 treeArr["py"][othermask]**2 - treeArr["pz"][othermask]**2)**
                0.5,
                axis=1,
            )

            if self.sqrt_counts:
                treeArr["nP"] = np.sqrt(treeArr["nP"])
                treeArr["nN"] = np.sqrt(treeArr["nN"])
                treeArr["nipip"] = np.sqrt(treeArr["nipim"])
                treeArr["nipi0"] = np.sqrt(treeArr["nipi0"])
                treeArr["niem"] = np.sqrt(treeArr["niem"])

            # One hot encoding of nutype
            treeArr["isNu"] = treeArr["PDGnu"] > 0
            treeArr["isNue"] = abs(treeArr["PDGnu"]) == 12
            treeArr["isNumu"] = abs(treeArr["PDGnu"]) == 14
            treeArr["isNutau"] = abs(treeArr["PDGnu"]) == 16

            # Get Weights
            weights = ak.to_numpy(treeArr['Weight'])
            # Convert to float32
            treeArr = ak.values_astype(treeArr[variables_out], np.float32)

            data = ak.to_numpy(treeArr)
            data = data.view(np.float32).reshape(
                (len(data), len(variables_out)))

            return data, np.expand_dims(weights, axis=1)