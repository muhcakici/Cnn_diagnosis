import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %%
data = pd.read_csv("data_diagnosis.csv")

# %%
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)

# %%
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"],axis=1)

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = scaler.fit_transform(x_data)


from keras.utils import to_categorical
Y = to_categorical(y, dtype ="uint8")

from sklearn.model_selection import train_test_split
trainX, testX, trainy, testy = train_test_split(x,Y,test_size = 0.2,random_state=42)
#%%

trainX = trainX.reshape(trainX.shape[0],trainX.shape[1], 1)
testX = testX.reshape(testX.shape[0],testX.shape[1], 1)


# from keras import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout,BatchNormalization

from keras import layers

from keras.models import Sequential # yazılması gerekli

verbose, epochs, batch_size = 0, 20, 8
n_features, n_outputs =trainX.shape[1], trainy.shape[1]

model = Sequential()
input_shape=(trainX.shape[1],1)
model.add(layers.Conv1D(filters=8, kernel_size=5, activation='relu', input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=3))
model.add(layers.Conv1D(filters=16, kernel_size=5, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(200, activation='relu'))
model.add(layers.Dense(n_outputs, activation='softmax'))
model.summary()
print("basladı")

import keras
import tensorflow
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss = keras.losses.categorical_crossentropy,
             optimizer = tensorflow.keras.optimizers.Adam(),
             metrics = ['accuracy'])
model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=1)
# evaluate model
_, accuracy = model.evaluate(testX, testy, verbose=0)


print(accuracy)