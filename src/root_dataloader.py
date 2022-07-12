import uproot
import awkward as ak
import numpy as np

from sklearn.utils import shuffle
from glob import glob
from torch.utils.data import Dataset


class ROOTDataset(Dataset):
    """
    Loads ROOT files of PDG types 12, -12, 14, -14 from two generators into memory.

    """

    def __init__(self,
                 data_dir,
                 generator_a='GENIEv2',
                 generator_b='NUWRO',
                 shuffle=True):
        super(ROOTDataset, self).__init__()
        self.data_dir = data_dir
        self.data = self.load_data(generator_a, generator_b).astype('float64')

    def __getitem__(self, index):
        features, label = self.data[index][:-1], self.data[index][-1]
        return {
            'features': features,
            'label': label
        }

    def __len__(self):
        return len(self.data)

    def load_data(self, generator_a, generator_b):
        dataset_a = self.load_generator(generator_a)
        dataset_b = self.load_generator(generator_b)
        data = np.vstack([dataset_a, dataset_b])
        if shuffle:
            np.random.shuffle(data)
        return data


    def load_generator(self, generator_name):
        wildcard_pattern = self.get_wildcard_pattern(generator_name)
        filenames = glob(self.data_dir + wildcard_pattern)
        data = self._rootfile_to_array(filenames[0])

        for filename in filenames[1:]:
            data = np.append(data, self._rootfile_to_array(filename), axis=0)

        # Create Labels
        if 'GENIE' in filename:
            labels = np.zeros(len(data))
        else:
            labels = np.ones(len(data))
        labels = np.expand_dims(labels, axis=1)

        return np.hstack([data, labels])

    def get_wildcard_pattern(self, generator_name):
        num_files = 50
        # String formatting to get the filename in the correct way
        index = np.random.randint(0, num_files)
        formatted_index = str(index).zfill(3)
        wildcard_pattern = f"*{generator_name}_1M_{formatted_index}_NUISFLAT.root"
        return wildcard_pattern


    def _rootfile_to_array(self, filename):
        variables_in, variables_out, m = self._get_constants()

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

            treeArr["nP"] = ak.count_nonzero(Pmask, axis=1)
            treeArr["nN"] = ak.count_nonzero(Nmask, axis=1)
            treeArr["nipip"] = ak.count_nonzero(Pipmask, axis=1)
            treeArr["nipim"] = ak.count_nonzero(Pimmask, axis=1)
            treeArr["nipi0"] = ak.count_nonzero(Pi0mask, axis=1)
            treeArr["nikp"] = ak.count_nonzero(Kpmask, axis=1)
            treeArr["nikm"] = ak.count_nonzero(Kmmask, axis=1)
            treeArr["nik0"] = ak.count_nonzero(K0mask, axis=1)
            treeArr["niem"] = ak.count_nonzero(EMmask, axis=1)

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

            treeArr["isNu"] = treeArr["PDGnu"] > 0
            treeArr["isNue"] = abs(treeArr["PDGnu"]) == 12
            treeArr["isNumu"] = abs(treeArr["PDGnu"]) == 14
            treeArr["isNutau"] = abs(treeArr["PDGnu"]) == 16

            treeArr = ak.values_astype(treeArr[variables_out], np.float32)

            data = ak.to_numpy(treeArr)
            data = data.view(np.float32).reshape(
                (len(data), len(variables_out)))
            return data

    def _get_constants(self):
        variables_in = [
            "cc",
            "PDGnu",
            "Enu_true",
            "ELep",
            "CosLep",
            "Q2",
            "W",
            "x",
            "y",
            "nfsp",
            "pdg",
            "E",
            "px",
            "py",
            "pz",
        ]

        variables_out = [
            "isNu",
            "isNue",
            "isNumu",
            "isNutau",
            "cc",
            "Enu_true",
            "ELep",
            "CosLep",
            "Q2",
            "W",
            "x",
            "y",
            "nP",
            "nN",
            "nipip",
            "nipim",
            "nipi0",
            "niem",
            "eP",
            "eN",
            "ePip",
            "ePim",
            "ePi0",
        ]
        m = {
            "P": 0.93827,
            "N": 0.93957,
            "piC": 0.13957,
            "pi0": 0.13498,
            "kC": 0.49368,
            "k0": 0.49764,
        }

        return variables_in, variables_out, m
