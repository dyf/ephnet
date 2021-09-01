import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

def build_inputs(pre_size):
    input_pre_i = tfk.Input(shape=(pre_size,), name='in_pre_i')
    input_pre_v = tfk.Input(shape=(pre_size,), name='in_pre_v')

    return [input_pre_i, input_pre_v]

def build_mlp(pre_size):
    inputs = build_inputs(pre_size)

    x = tfkl.Concatenate()(inputs)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(200, activation='relu', name='d0')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(150, activation='relu', name='d1')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(100, activation='relu', name='d2')(x)
    x = tfkl.BatchNormalization()(x)
    x = tfkl.Dense(1, activation='linear', name='out_v')(x)

    return tfk.Model(inputs=inputs, outputs=x)

def build_cnn(size):
    inp = tfk.Input(shape=(size,2))
            
    x = tfkl.Conv1D(256, kernel_size=9, padding='causal', activation='relu')(inp)
    x = tfkl.MaxPooling1D()(x)
    x = tfkl.Conv1D(128, kernel_size=9, padding='causal', activation='relu')(x)
    x = tfkl.MaxPooling1D()(x)
    x = tfkl.Conv1D(128, kernel_size=9, padding='causal', activation='relu')(x)
    x = tfkl.MaxPooling1D()(x)            
    x = tfkl.Reshape(target_shape=(x.shape[1]*x.shape[2],))(x)
    x = tfkl.Dense(128, activation='relu')(x)
    x = tfkl.Dense(1, activation='linear')(x)    
    
    return tfk.Model(inputs=inp, outputs=x)

if __name__ == "__main__": 
    m = build_cnn(1000, 4000)
    print(m.summary())