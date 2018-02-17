import numpy as np
import pandas as pd
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers import Conv2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


columns = ['Center Image', 'Left Image', 'Right Image', 'Steering', 'Throttle', 'Brake', 'Speed']

df = pd.read_csv('data/driving_log.csv', header=None, names=columns)
df = pd.DataFrame(df)

images = []
for line in df['Center Image']:
    image = cv2.imread(line)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)

print(len(images))

measurements = []
for line in df['Steering']:
    measurement = float(line)
    measurements.append(measurement)

print(len(measurements))

#convert to numpy array
X = np.array(images)
y = np.array(measurements)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)

#Preprocessing
X_normalized = np.array(X_train / 255.0 - 0.5 )
print("normalized")

def Nvidia_model():
    model = Sequential()
    keep_prob = 0.5

    #First ConLayer
    model.add(Conv2D(24,(5, 5), strides=(2,2), padding='valid',activation='relu',input_shape=(160,320,3)))

    #second ConLayer
    model.add(Conv2D(36,(5, 5), strides=(2,2), padding='valid',activation='relu'))

    #third ConLayer
    model.add(Conv2D(48,(5, 5), strides=(2,2), padding='valid',activation='relu'))

    #fourth ConLayer
    model.add(Conv2D(64,(3, 3),  padding='valid',activation='relu'))

    #fifth ConLayer
    model.add(Conv2D(64,(3, 3), padding='valid',activation='relu'))

    #Flatten
    model.add(Flatten())

    #FullConnectedLayer 1
    model.add(Dense(100,activation='relu'))
    model.add(Dropout(keep_prob))

    #FullConnectedLayer 2
    model.add(Dense(50,activation='relu'))
    model.add(Dropout(keep_prob))

    #FullConnectedLayer 3
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(keep_prob))

    #Output Layer
    model.add(Dense(1, activation='softsign'))

    return model

#Train the model
epochs = 10
batch_size = 128

model = Nvidia_model()
model.compile(loss = 'mse', optimizer='adam')
print("start training")
model.fit(X_normalized,y_train, batch_size=batch_size, validation_split=0.2, shuffle=True,epochs=epochs)

#Save the model
model.save('model.h5')
print("model saved")



