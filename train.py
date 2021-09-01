import tensorflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt
import numpy as np
import os

import data as mdata
import model as mmodel

def viz(i, v, model, size, t0, t1):
    new_v = np.zeros_like(v)
    new_v[t0-size:t0] = v[t0-size:t0]

    for ii in range(t0,t1):
        di = np.stack([i[ii-size:ii], new_v[ii-size:ii]], axis=-1)
        di = np.reshape(di, (1, di.shape[0], di.shape[1]))
        new_v[ii] = model.predict(di)[0]

    t = np.arange(len(i)) / 1.0e6
    plt.plot(t[t0-size:t1],i[t0-size:t1])
    plt.plot(t[t0-size:t1],v[t0-size:t1])
    plt.plot(t[t0-size:t1],new_v[t0-size:t1])


def train(size, batch_size, n_epochs, lr, data_file_name, metadata_file_name, output_dir):
    data = mdata.EphysData(
        data_file_name = data_file_name,
        metadata_file_name = metadata_file_name
    )

    viz_data = data.read_one(index=20000)
    
    #model = mmodel.build_mlp(pre_size)
    model = mmodel.build_cnn(size)
    optimizer = tf.keras.optimizers.Adam(lr)

    model.compile(loss="mse", optimizer=optimizer)

    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, output_dir, max_to_keep=3)
    manager.restore_or_initialize()
    
    ct = 0
    for i in range(n_epochs):
        md = data.metadata.sample(frac=1)
    
        for j,row in md.iterrows():
            d = data.read_one(j)
            
            ds = tf.keras.preprocessing.sequence.TimeseriesGenerator(
                data=d, 
                targets=d[:,1], 
                length=size,
                stride=100, 
                batch_size=10
            )
            print(f"{row.dataset_name}")        

            res = model.fit(ds, verbose=2)            
            
            ct += 1

            if ct % 10000 == 0:
                viz(viz_data[:,0], viz_data[:,1], model, size, 10000, 30000)
                plt.savefig(os.path.join(output_dir, f'debug_{i}_{ct:07d}.png'))
                plt.close()
                print(f"epoch {i+1}/{n_epochs}, finished {j+1}/{len(md)} sweeps")
                manager.save()

if __name__ == "__main__":
    train(**{
        'batch_size': 128,
        'n_epochs': 50, 
        'size': 1000,
        'lr': 0.001,
        'data_file_name': 'train_data/recordings.h5',
        'metadata_file_name': 'train_data/metadata.csv',
        'output_dir': './train_output'
    })