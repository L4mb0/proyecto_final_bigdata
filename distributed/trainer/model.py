import tensorflow as tf
from tensorflow import keras
from tensorflow import data
from data import Dataset
from keras.layers import Dense,Dropout,Input,Flatten
from keras.models import Model
from keras.applications.mobilenet_v2 import MobileNetV2

# number_of_classes parameter not use
def create_model():
    mobilenet = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(160,160,3)))
    
    mobilenet.trainable = False
    output = Flatten()(mobilenet.output)
    output = Dropout(0.3)(output)
    output = Dense(units = 8,activation='relu')(output)
    prediction = Dense(1,activation='sigmoid')(output)
    
    model = Model(inputs = mobilenet.input,outputs = prediction)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.000001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        metrics=['accuracy']
    )
    
    return model