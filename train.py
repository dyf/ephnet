import tensorflow as tf
import tensorflow.keras as tfk

import data as mdata
import model as mmodel

class CustomCallback(tfk.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        manager.save()

data = mdata.EphysData(
    data_file_name="train_data/recordings.h5",
    metadata_file_name = "train_data/metadata.csv"
)

ds = tf.data.Dataset.from_generator(
    generator=lambda: data.iter(1000), 
    output_signature=(
        ( 
            tf.TensorSpec(shape=(1000,), dtype=tf.float32), 
            tf.TensorSpec(shape=(1000,), dtype=tf.float32),
            tf.TensorSpec(shape=(4000,), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(4000,), dtype=tf.float32)
    )
)

model = mmodel.build_cnn(1000,4000)
optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(loss="mse", optimizer=optimizer, metrics=["mean_squared_error"])

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, './train_output', max_to_keep=3)
manager.restore_or_initialize()

model.fit(ds.batch(64), epochs=50, callbacks=[CustomCallback()])