import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def build_model(pre_size, size):
    input_pre_i = tfk.Input(shape=(pre_size,), name='in_pre_i')
    input_v_i = tfk.Input(shape=(pre_size,), name='in_pre_v')
    input_v = tfk.Input(shape=(size,), name='in_i')

    x = tfkl.Concatenate()([input_pre_i, input_v_i, input_v])
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(300, activation='relu', name='d0')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(100, activation='relu', name='d1')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(50, activation='relu', name='d2')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(size, activation='linear', name='out_v')(x)

    return tfk.Model(inputs=[input_pre_i, input_v_i, input_v], outputs=x)

if __name__ == "__main__": 
    m = build_model(1000, 4000)
    print(m.summary())