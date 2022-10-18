import glob

import numpy as np

from utils.funcs import rootfile_to_array


def format_filename(generator_name, index):
        """
        Creates a ROOT filename wildcard with the passed index
        """
        formatted_index = str(index).zfill(3)
        filename_pattern = f"*{generator_name}*_1M_{formatted_index}_NUISFLAT.root"
        return filename_pattern

root_dir = '/eos/home-c/cristova/DUNE/AlternateGenerators/'
out_dir = '/eos/user/r/rradev/generator_reweigthing/'
generators = [
    # 'GENIEv2',
    # 'GENIEv3_G18_10b', 
    'GENIEv3_G18_10a',
    'NUWRO',
    'NEUT']

for generator_name in generators:
    for index in range(50):
        filenames = glob.glob(root_dir + format_filename(generator_name, index))
        out_filename = f'{out_dir}{generator_name}{index}'
        # Check if we have all 4 PDG types
        if len(filenames) == 4:
            print(filenames)
            # Create data to be saved
            try:
                data, weights = rootfile_to_array(filenames[0])
                for filename in filenames[1:]:
                    new_data, new_weights = rootfile_to_array(filename)
                    data = np.append(data, new_data, axis=0)
                
                np.save(out_filename, data)
            except:
                pass
