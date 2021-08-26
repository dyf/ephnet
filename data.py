import pandas as pd
import numpy as np
import h5py 
import scipy.signal

np.random.seed(1)

def read_sweep(data, sweep_num):
    sweep_data = data.get_sweep(sweep_num)

    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][index_range[0]:index_range[1]+1] # in A
    v = sweep_data["response"][index_range[0]:index_range[1]+1] # in V
    i *= 1e12 # to pA
    v *= 1e3 # to mV
    t = np.arange(len(v)) * (1.0 / sweep_data["sampling_rate"])

    return i,v,t,sweep_data["sampling_rate"]

def resample_sweep(i, v, t, from_hz, to_hz):
    ratio = from_hz / to_hz
    iratio = int(ratio)

    i = scipy.signal.medfilt(i, kernel_size=iratio)
    v = scipy.signal.medfilt(v, kernel_size=iratio)

    return i[::iratio],v[::iratio],t[::iratio]

def sample_sweep(i, v, sample_size):
    idx = np.arange(len(i))
    stim_on = np.where(i != 0)[0]
    
    if len(stim_on) == 0: 
        raise IndexError("No stimulus")
    
    start = max(0, min(stim_on) - sample_size//2)
    stop = min(len(v)-sample_size-1, max(stim_on) - sample_size//2)
    idx = np.random.randint(low=start,high=stop)
    return i[idx:idx+sample_size], v[idx:idx+sample_size]
    


def filter_cells(cells):
    cells = cells[cells.species == "Mus musculus"]
    cells = cells[cells.reporter_status == "positive"]
    
    return cells

def filter_sweep_table(sweep_table):
    return sweep_table[sweep_table.aibs_stimulus_name != 'Test']

def build_sweep_table(data):
    sweeps = []
    for sn in data.get_experiment_sweep_numbers():
        md = data.get_sweep_metadata(sn)
        md['sweep_number'] = sn
        sweeps.append(md)
    return pd.DataFrame.from_records(sweeps, index='sweep_number')

def main():
    import allensdk.core.cell_types_cache as ctc

    target_sampling_rate = 10000 # khz
    target_sample_size = int(500 * 1.0e-3 * target_sampling_rate) # 500ms
    number_of_samples = 500000    
    output_data_file_name = "train_data/recordings.h5"
    output_metadata_file_name = "train_data/metadata.csv"

    output_metadata = []    
    
    cache = ctc.CellTypesCache(manifest_file='ctc/manifest.json')
    cells = cache.get_cells()
    cells = pd.DataFrame.from_records(cells)
    
    cells = filter_cells(cells)

    samples_per_cell = number_of_samples // len(cells)
    number_of_samples = samples_per_cell * len(cells)

    with h5py.File(output_data_file_name, "w") as hf:
        ds = hf.create_dataset("recordings", shape=(number_of_samples, target_sample_size, 2), dtype=float, fillvalue=0.0)

        si = 0
        for ci, row in cells.iterrows():
            cell_id = row.id

            cell_data = cache.get_ephys_data(cell_id)
            sweep_table = build_sweep_table(cell_data)
            sweep_table = filter_sweep_table(sweep_table)
            
            samples_per_sweep = samples_per_cell // len(sweep_table) 

            print(len(cells), samples_per_cell, samples_per_sweep)
            
            for sweep_num in sweep_table.index:
                i,v,t,s = read_sweep(cell_data, sweep_num)
                i,v,t = resample_sweep(i, v, t, s, target_sampling_rate)

                for sweep_i in range(samples_per_sweep):
                    try:
                        sweep_i, sweep_v = sample_sweep(i, v, target_sample_size)
                    except IndexError as e:
                        print(f"skipping sweep {sweep_i}")
                        continue

                    ds[si,:,0] = sweep_i
                    ds[si,:,1] = sweep_v

                    si += 1

                    output_metadata.append({
                        'sweep_number': sweep_num,
                        'cell_id': cell_id,
                        'area': row.structure_area_abbrev,
                        'hemisphere': row.structure_hemisphere,
                        'transgenic_line': row.transgenic_line,
                        'layer': row.structure_layer_name
                    })
            
                    if si % 1000 == 0:
                        print(si)


    pd.DataFrame.from_records(output_metadata).to_csv(output_metadata_file_name)
            
class EphysData:
    def __init__(self, metadata_file_name, data_file_name):
        self.metadata_file_name = metadata_file_name
        self.data_file_name = data_file_name

        self.metadata = pd.read_csv(metadata_file_name, index_col=0)    

    def read_one(self, index, pre_size):
        with h5py.File(self.data_file_name,"r") as f:
            d = f["recordings"][index]
            return {
                "pre_i": d[:pre_size,0], 
                "pre_v": d[:pre_size,1],
                "i": d[pre_size:,0],
                "v": d[pre_size:,1]
            }

    def iter(self, pre_size):
        N = len(self.metadata)
        with h5py.File(self.data_file_name,"r") as f:
            d = f["recordings"]
            print(d.shape)
            
            for i in range(N):
                x = d[i]
                yield ( 
                    ( 
                        x[:pre_size,0], 
                        x[:pre_size,1], 
                        x[pre_size:,0] 
                    ), 
                    x[pre_size:,1] 
                )

if __name__=="__main__": main()