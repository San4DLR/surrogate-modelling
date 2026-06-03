import json
import numpy as np


def getInpsAndOuts(ROOT_DATA_PATH):
    all_inps = []
    all_outs = []

    with open(ROOT_DATA_PATH / 'names.json') as f:
        samples = json.load(f)

    for sample in samples:
        input_file_name = f"{sample}_geometry_inputs.json"
        with open(ROOT_DATA_PATH / input_file_name) as f:
            all_inps += [json.load(f)]
        
        output_file_name = f"{sample}_pressure.npy"
        all_outs += [np.load(ROOT_DATA_PATH / output_file_name)]

    all_outs = np.array(all_outs)

    ### hard coding now...

    common_keys = ['points_halfshape', 'plotmode', 'extension_ribs_front', 'runDuctGen', 
                'mode', 'extension_ribs_aft', 'extension_aft', 
                'rel_deriv3_startstop_min', 'extension_front']

    rectangular = ['inlet_x', 'inlet_y', 'inlet_z', 'aip_x', 'aip_y', 'aip_z', 'aip_r', 'inlet_area', 
                    'rotation', 'n_in', 'a', 'b', 'c', 
                    's_curvature_start', 's_curvature_end', 's_diffusion_start', 's_diffusion_end', 
                    's_morph_start', 's_morph_end']


    filtered_inps = np.empty((len(all_inps), len(rectangular)))

    for batch, inps in enumerate(all_inps):
        for i, key in enumerate(rectangular):
            filtered_inps[batch, i] = inps['rectangular_duct'][key]

    return filtered_inps, all_outs

if __name__ == "__main__":

    ROOT_DATA_PATH = 'C:\\Users\\pisk_sa\\Code\\surrogate-modelling\\data\\duct_pressure_distribution\\N128'
    
    inps, outs = getInpsAndOuts(ROOT_DATA_PATH)