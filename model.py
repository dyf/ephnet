import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def build_inputs(pre_size, size):
    input_pre_i = tfk.Input(shape=(pre_size,), name='in_pre_i')
    input_pre_v = tfk.Input(shape=(pre_size,), name='in_pre_v')
    input_v = tfk.Input(shape=(size,), name='in_i')

    return [input_pre_i, input_pre_v, input_v]

def build_mlp(pre_size, size):
    inputs = build_inputs(pre_size, size)

    x = tfkl.Concatenate()(inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(200, activation='relu', name='d0')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(150, activation='relu', name='d1')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(100, activation='relu', name='d2')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(size, activation='linear', name='out_v')(x)

    return tfk.Model(inputs=inputs, outputs=x)

def build_cnn(pre_size, size):
    inputs = build_inputs(pre_size, size)
    [pre_i, pre_v, v] = inputs
        
    pre_i = tfkl.Reshape(target_shape=(pre_i.shape[1], 1))(pre_i)
    pre_i = tfkl.BatchNormalization()(pre_i)
    pre_v = tfkl.Reshape(target_shape=(pre_v.shape[1], 1))(pre_v)
    pre_v = tfkl.BatchNormalization()(pre_v)
    pre = tfkl.Concatenate(axis=2)([pre_i, pre_v])

    pre = tfkl.Conv1D(100, kernel_size=10, padding='causal', activation='relu')(pre)
    pre = tfkl.UpSampling1D(2)(pre)
    pre = tfkl.Conv1D(100, kernel_size=10, padding='causal', activation='relu')(pre)
    pre = tfkl.UpSampling1D(2)(pre)
    
    v = tfkl.Reshape(target_shape=(v.shape[1], 1))(v)
    v = tfkl.BatchNormalization()(v)
    #v = tfkl.Conv1D(100, kernel_size=10, padding='causal', activation='relu')(v)

    x = tfkl.Concatenate(axis=2)([pre, v])    
    x = tfkl.Conv1D(100, kernel_size=10, padding='causal', activation='relu')(x)
    x = tfkl.Conv1D(100, kernel_size=10, padding='causal', activation='relu')(x)
    
    x = tfkl.Conv1D(1, kernel_size=10, padding='causal', activation='linear')(x)
    x = tfkl.Reshape(target_shape=(size,))(x)
    
    return tfk.Model(inputs=inputs, outputs=x)

if __name__ == "__main__": 
    m = build_cnn(1000, 4000)
    print(m.summary())