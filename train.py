import tensorflow as tf
import tensorflow.keras as tfk

import data as mdata
import model as mmodel

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

checkpoint_filepath = 'train_output/checkpoint-{epoch:02d}-{loss:.2f}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='loss',
    save_weights_only=True,
    save_best_only=True,
    save_freq='epoch')


model = mmodel.build_model(1000,4000)
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

model.fit(ds.batch(16), epochs=10, callbacks=[model_checkpoint_callback])