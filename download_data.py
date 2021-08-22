import allensdk.core.cell_types_cache as ctc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plot_sweep(sweep_data):
    index_range = sweep_data["index_range"]
    i = sweep_data["stimulus"][0:index_range[1]+1] # in A
    v = sweep_data["response"][0:index_range[1]+1] # in V
    i *= 1e12 # to pA
    v *= 1e3 # to mV

    sampling_rate = sweep_data["sampling_rate"] # in Hz
    t = np.arange(0, len(v)) * (1.0 / sampling_rate)

    #plt.style.use('ggplot')
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(t, v, color='black')
    axes[1].plot(t, i, color='gray')
    axes[0].set_ylabel("mV")
    axes[1].set_ylabel("pA")
    axes[1].set_xlabel("seconds")
    plt.show()

def main():
    target_sampling_rate = 10000 # khz
    pre_size = 100 * int(target_sampling_rate * 1e-3)
    size = 400 * int(target_sampling_rate * 1e-3)
    print(pre_size, size)
    return

    n_cells = 2
    cache = ctc.CellTypesCache(manifest_file='ctc/manifest.json')
    cells = cache.get_cells(species=['Mus musculus'])

    cells = cells[:n_cells]
    cells = pd.DataFrame.from_records(cells)
    for cell_id in cells.id:
        ephys = cache.get_ephys_data(cell_id)
        sweep_nums = ephys.get_experiment_sweep_numbers()
        
        sweeps = []
        for sn in sweep_nums:
            sweeps.append(ephys.get_sweep_metadata(sn))            
        
        sweeps = pd.DataFrame.from_records(sweeps)
        sweeps = sweeps[sweeps.aibs_stimulus_name == "Long Square"]
        print(sweeps)

        sweep_data = ephys.get_sweep(11)
        plot_sweep(sweep_data)
        



if __name__=="__main__": main()