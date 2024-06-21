# use code from https://github.com/CaloChallenge/homepage/blob/main/code/HighLevelFeatures.ipynb
import numpy as np
from argparse import ArgumentParser

from funcx import FuncXExecutor

np.set_printoptions(suppress=True)

def apply_mask(mask, X_train, input_file, add_noise=False):
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    
    from training.common import get_energies
    from training.common import split_energy
    from training.common import plot_frame
    
    np.seterr(divide = 'ignore', invalid='ignore')
    event_energy_before = X_train.sum(axis=1)[:]
    if add_noise:
        # X_train is in MeV, add uniform noise of [0, 0.1keV]
        X_train += np.random.uniform(low=0, high=0.0001, size=X_train.shape)
        print('\033[92m[INFO] Add noise\033[0m', 0, 0.1, '[keV] for voxel energy')
    event_energy_before2 = X_train.sum(axis=1)[:]

    # mask too low energy to zeros
    if isinstance(mask, (int, float)):
        X_train[X_train < (mask / 1000)] = 0
    elif isinstance(mask, dict):
        # X_train is un-sorted!
        energies = get_energies(input_file)
        for k,m in mask.items():
            X_train[np.logical_and(energies == k, X_train < (m / 1000))] = 0
    else:
        raise NotImplementedError

    # plot energy change before and after masking
    event_energy_after  = X_train.sum(axis=1)[:]
    event_energy = np.concatenate([event_energy_before.reshape(-1,1), event_energy_before2.reshape(-1,1), event_energy_after.reshape(-1,1)], axis=1)

    categories, vector_list  = split_energy(input_file, event_energy)
    fig, axes = plot_frame(categories, xlabel="Rel. change in E total", ylabel="Events")
    for index, energy in enumerate(categories):
        ax = axes[index]
        before, after = vector_list[index][:,0], vector_list[index][:,-1]
        x = 1 - np.divide(after, before, out=np.zeros_like(before), where=before!=0)
        if x.max() < 1E-4:
            high = 1E-4
        elif x.max() < 1E-3:
            high = 1E-3
        elif x.max() < 1E-2:
            high = 1E-2
        elif x.max() < 0.1:
            high = 0.1
        else:
            high = 1
        n, _, _ = ax.hist(x, bins=100, range=(0,high))
        ax.set_yscale('symlog')
        ax.set_ylim(bottom=0)
        if isinstance(mask, (int, float)):
            mask_legend = f'Mask {mask} keV\nMax {high}'
        elif isinstance(mask, dict):
            if mask[energy] < 1E3:
                mask_legend = f'Mask {mask[energy]} keV\nMax {high}'
            elif mask[energy] < 1E6:
                mask_legend = f'Mask {mask[energy]/1E3} MeV\nMax {high}'
            else:
                mask_legend = f'Mask {mask[energy]/1E6} GeV\nMax {high}'
        ax.text(0.98, 0.88, mask_legend, transform=ax.transAxes, va="top", ha="right", fontsize=15)
    ax = axes[-1]
    ax.axis("on")
    x = 1 - event_energy_after / event_energy_before
    if x.max() < 1E-4:
        high = 1E-4
    elif x.max() < 1E-3:
        high = 1E-3
    elif x.max() < 1E-2:
        high = 1E-2
    elif x.max() < 0.1:
        high = 0.1
    else:
        high = 1
    ax.hist(x, bins=100, range=(0,high))
    ax.set_yscale('symlog')
    ax.set_ylim(bottom=0)
    os.makedirs(args.output_path, exist_ok=True)
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    plt.savefig(os.path.join(args.output_path, f'mask_{particle}_{args.mask}keV.pdf'))
    print('\033[92m[INFO] Mask\033[0m', args.mask, mask, '[keV] for voxel energy')
        
    # return masked input
    return X_train

def main(args, model):
    #from HighLevelFeatures import HighLevelFeatures
    import numpy as np
    import h5py, os, json
    import matplotlib.pyplot as plt
    from pdb import set_trace
    from training.common import get_energies
    from training.common import get_kin
    from training.common import kin_to_label
    from training.data import preprocessing
    from training.model import WGANGP
    from training.train import plot_input
    #from data import *
    import re

    # creating instance of HighLevelFeatures class to handle geometry based on binning file
    input_file = args.input_file
    particle = input_file.split('/')[-1].split('_')[-2][:-1]
    #hlf = HighLevelFeatures(particle, filename=f'{os.path.dirname(input_file)}/binning_dataset_1_{particle}s.xml')
    print('\033[92m[INFO] Run\033[0m', particle, input_file)
    
    # loading the .hdf5 datasets
    photon_file = h5py.File(f'{input_file}', 'r')
    
    energies = get_energies(input_file)
    kin, particle = get_kin(input_file)
    label_kin = kin_to_label(kin)
    
    X_train = photon_file['showers'][:]
    if args.mask is not None:
        if args.mask < 0:
            mask = list(np.unique(energies)/256 * abs(args.mask)) # E/256 * (-mask)
            mask = dict(zip(list(np.unique(energies)), mask))
        else:
            mask = args.mask
        X_train = apply_mask(mask, X_train, input_file, add_noise=args.add_noise)

    if args.preprocess is not None:
        if (re.compile("^log10.([0-9.]+)+$").match(args.preprocess) \
                or re.compile("^scale.([0-9.]+)+$").match(args.preprocess) \
                or re.compile("^slope.([0-9.]+)+$").match(args.preprocess)
            ): # log10.x, scale.x, slope
            X_train, scale = preprocessing(X_train, kin, name=args.preprocess, input_file=input_file)
    else:
        X_train = preprocessing(X_train, kin, name=args.preprocess, input_file=input_file)
        scale = None

    if 'photon' in particle:
        hp_config = {
            'model': model if model else 'BNswish',
            'G_size': 1,
            'D_size': 1,
            'optimizer': 'adam',
            'G_lr': 1E-4,
            'D_lr': 1E-4,
            'G_beta1': 0.5,
            'G_beta1': 0.5,
            'batchsize': 1024,
            'datasize': X_train.shape[0],
            'dgratio': 8,
            'latent_dim': 50,
            'lam': 3,
            'conditional_dim': label_kin.shape[1],
            'generatorLayers': [50, 100, 200],
            'nvoxels': X_train.shape[1],
            'use_bias': True,
        }
    else: # pion
        hp_config = {
            'model': model if model else 'noBN',
            'G_size': 1,
            'D_size': 1,
            'optimizer': 'adam',
            'G_lr': 1E-4,
            'D_lr': 1E-4,
            'G_beta1': 0.5,
            'G_beta1': 0.5,
            'batchsize': 1024,
            'dgratio': 5,
            'latent_dim': 50,
            'lam': 10,
            'conditional_dim': label_kin.shape[1],
            'generatorLayers': [50, 100, 200],
            'discriminatorLayers': [800, 400, 200],
            'nvoxels': X_train.shape[1],
            'use_bias': True,
            'preprocess': args.preprocess,
        }
    if args.config:
        from quickstats.utils.common_utils import combine_dict
        hp_config = combine_dict(hp_config, json.load(open(args.config, 'r')))

    job_config = {
        'particle': particle+'s',
        'eta_slice': '20_25',
        'checkpoint_interval': 1 if args.example_run else 1000 if not args.debug else 10,
        'output': args.output_path,
        'max_iter': 2 if args.example_run else 4E5 if args.loading else 1E6,
        'cache': False,
        'loading': args.loading,
    }

    wgan = WGANGP(job_config=job_config, hp_config=hp_config, logger=__file__)
    if scale:
        with open(f'{wgan.train_folder}/scale_{args.preprocess}.json', 'w') as fp:
            json.dump(scale, fp, indent=2)
    plot_input(args, X_train, output=wgan.train_folder)
    wgan.train(X_train, label_kin)


def plot_input(args, X_train, output):
    import os
    
    from training.common import get_kin
    from training.common import split_energy
    from training.common import plot_energy_vox
    
    kin, particle = get_kin(args.input_file)
    categories, xtrain_list = split_energy(args.input_file, X_train)
    out_file = os.path.join(output, f'input_{particle}_{args.preprocess}.pdf')
    plot_energy_vox(categories, [xtrain_list], label_list=['Input'], nvox='all', logx=False, \
            particle=particle, output=out_file, draw_ref=False, xlabel='Energy of voxel as training input [MeV]')
    print('\033[92m[INFO] Save to\033[0m', out_file)

if __name__ == '__main__':
    import time
    
    all_models = [ 'GANv1', 'BNReLU', 'BNswish', 'BNswishHe', 'BNLeakyReLU', 'noBN', 'SN' ]

    """Get arguments from command line."""
    parser = ArgumentParser(description="\033[92mConfig for training.\033[0m")
    parser.add_argument('-i', '--input_file', type=str, required=False, default='', help='Training h5 file name (default: %(default)s)')
    parser.add_argument('-o', '--output_path', type=str, required=True, default='../output/dataset1/v1', help='Training h5 file path (default: %(default)s)')
    parser.add_argument('-c', '--config', type=str, required=False, default=None, help='External config file (default: %(default)s)')
    parser.add_argument('-m', '--models', nargs='*', required=False, default=all_models, choices=all_models, help='Models to train (default: %(default)s)')
    parser.add_argument('--mask', type=float, required=False, default=None, help='Mask low noisy voxels in keV (default: %(default)s)')
    parser.add_argument('--debug', required=False, action='store_true', help='Debug mode (default: %(default)s)')
    parser.add_argument('-p', '--preprocess', type=str, required=False, default=None, help='Preprocessing name (default: %(default)s)')
    parser.add_argument('-l', '--loading', type=str, required=False, default=None, help='Load model (default: %(default)s)')
    parser.add_argument('--add_noise', required=False, action='store_true', help='Add noise (default: %(default)s)')
    parser.add_argument('--endpoint-id', required=False, type=str, default=None, help='FuncX Endpoint ID')
    parser.add_argument('--example-run', required=False, action="store_true", default=False, help='Run with only 2 max intervals and 1 checkpoint interval')

    args = parser.parse_args()

    # example where we parallelize over the models
    
    if args.endpoint_id is not None: # if we execute with funcX
        results = []
        with FuncXExecutor(endpoint_id=args.endpoint_id) as fxe:
            for m in args.models: # Iterate through all the models and submit tasks for each model
                results.append(fxe.submit(main, args, m))

        _ = [r.result() for r in results] # Acquire the result
    else: # What execution would look like without funcX (i.e., sequential execution)
        for m in args.models:
            main(args, model=m)
