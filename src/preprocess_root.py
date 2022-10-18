import argparse
import os
from platform import mac_ver

import awkward as ak
import awkward0
import numpy as np
import uproot
import vector
from uproot3_methods import TLorentzVectorArray


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(axis=1)
    if eps is not None:
        m2 = np.clip(m2, a_min=eps, a_max=np.max(m2))
    return m2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(axis=1)
    if eps is not None:
        pt2 = np.clip(pt2, a_min=eps, a_max=np.max(pt2))
    return pt2


def to_ptrapphim(x, return_mass=True, eps=1e-8):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), axis=1)
    pt = np.sqrt(to_pt2(x, eps=eps))
    # raparticle_idity = 0.5 * torch.log((energy + pz) / (energy - pz))
    raparticle_idity = 0.5 * np.log(1 + (2 * pz) / (energy - pz))
    clip = np.clip(raparticle_idity, min=1e-20, max=np.max(raparticle_idity))
    phi = (np.arctan2)(py, px)
    if not return_mass:
        return np.cat((pt, raparticle_idity, phi), axis=1)
    else:
        m = np.sqrt(to_m2(x, eps=eps))
        return np.cat((pt, raparticle_idity, phi, m), axis=1)

'''
Datasets introduction:
https://energyflow.network/docs/datasets/#quark-and-gluon-jets

Download:
- Pythia8 Quark and Gluon Jets for Energy Flow:
  - https://zenodo.org/record/3164691

- Herwig7.1 Quark and Gluon Jets:
  - https://zenodo.org/record/3066475
'''


def _transform(sourcefile):
    # source_array: (num_data, max_num_particles, 4)
    # (pt,y,phi,particle_id)

    # with uproot.open(sourcefile) as file:
    file = uproot.open(sourcefile)
    tree = file['FlatTree_VARS']
    px = tree['px']
    py = tree['py']
    pz = tree['pz']
    energy = tree['E']
    particle_id = tree['pdg'].array()

    p4 = vector.zip({
        'px': px.array(),
        'py': py.array(),
        'pz': pz.array(), 
        'E': energy.array()
    })

    mask = energy.array() > 0
    n_particles = np.sum(mask, axis=1)

    px = p4.px
    py = p4.py
    pz = p4.pz
    energy = p4.E

    energy_sum = ak.sum(energy, axis=1)
    event_p4 = vector.zip({
        'px': ak.sum(px, axis=1),
        'py': ak.sum(py, axis=1),
        'pz': ak.sum(pz, axis=1),
        'E': energy_sum
    })

    # outputs
    v = {}
    # v['label'] = y

    v['event_pt'] = event_p4.pt
    v['event_eta'] = event_p4.eta
    v['event_phi'] = event_p4.phi
    v['event_energy'] = event_p4.energy
    v['event_mass'] = event_p4.mass
    v['event_nparticles'] = n_particles

    v['part_px'] = px
    v['part_py'] = py
    v['part_pz'] = pz
    v['part_energy'] = energy

    _event_etasign = np.sign(v['event_eta'])
    v['part_deta'] = (p4.eta - v['event_eta']) * _event_etasign
    v['part_dphi'] = p4.deltaphi(event_p4)

    v['part_particle_id'] = particle_id
    v['part_isCHPlus'] = ak.values_astype(((particle_id == 211) + (particle_id == 321) + (particle_id == 2212)), np.float32)
    v['part_isCHMinus'] = ak.values_astype(((particle_id == -211) + (particle_id == -321) + (particle_id == -2212)), np.float32)
    v['part_isNeutralHadron'] = ak.values_astype(((particle_id == 130) + (particle_id == 2112) + (particle_id == -2112)), np.float32)
    v['part_isPhoton'] = ak.values_astype((particle_id == 22), np.float32)
    v['part_isEPlus'] = ak.values_astype((particle_id == -11), np.float32)
    v['part_isEMinus'] = ak.values_astype((particle_id == 11), np.float32)
    v['part_isMuPlus'] = ak.values_astype((particle_id == -13), np.float32)
    v['part_isMuMinus'] = ak.values_astype((particle_id == 13), np.float32)

    v['part_isChargedHadron'] = v['part_isCHPlus'] + v['part_isCHMinus']
    v['part_isElectron'] = v['part_isEPlus'] + v['part_isEMinus']
    v['part_isMuon'] = v['part_isMuPlus'] + v['part_isMuMinus']

    v['part_charge'] = (v['part_isCHPlus'] + v['part_isEPlus'] + v['part_isMuPlus']
                        ) - (v['part_isCHMinus'] + v['part_isEMinus'] + v['part_isMuMinus'])

    for k in list(v.keys()):
        if k.endswith('Plus') or k.endswith('Minus'):
            del v[k]

    return v


def convert(sources, destdir, basename):
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    for idx, sourcefile in enumerate(sources):
        output = os.path.join(destdir, '%s_%d.parquet' % (basename, idx))
        print(sourcefile)
        print(output)
        if os.path.exists(output):
            os.remove(output)
        v = _transform(sourcefile)
        arr = ak.Array({k: ak.from_awkward0(a) for k, a in v.items()})
        ak.to_parquet(arr, output, compression='LZ4', compression_level=4)


def natural_sort(l):
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert ROOT datasets')
    parser.add_argument('-i', '--inputdir', default='/eos/home-c/cristova/DUNE/AlternateGenerators/', help='Directory of ROOT input Files.')
    parser.add_argument('-o', '--outputdir', default='/eos/user/r/rradev/particle_cloud/', help='Output directory.')
    parser.add_argument('--train-test-split', default=0.8, help='Training / testing split fraction.')
    args = parser.parse_args()

    import glob
    sources = natural_sort(glob.glob(os.path.join(args.inputdir, 'flat_argon_12_GENIEv3_G18_10b_00_000_1M_*_NUISFLAT.root')))
    print(sources)
    n_train = int(args.train_test_split * len(sources))
    train_sources = sources[:n_train]
    test_sources = sources[n_train:]

    convert(train_sources, destdir=args.outputdir, basename='train_file_flat_argon_12_GENIEv3_G18_10b')
    convert(test_sources, destdir=args.outputdir, basename='test_file_flat_argon_12_GENIEv3_G18_10b')