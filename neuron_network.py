import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.model_selection import train_test_split

train_data = pd.read_csv("digit-recognizer/train.csv")
test_data = pd.read_csv("digit-recognizer/test.csv")
y_train = train_data["label"]
y_test_final = test_data.values
n_train = train_data.shape[0]
m_train = train_data.shape[1]
n_test = test_data.shape[0]

train_data.drop(['label'], axis='columns', inplace=True)
train_data = train_data.values
y_train = y_train.values

X_validation = test_data.values.reshape(n_test,28,28,1)
X_train = train_data.reshape(n_train,28,28,1)
y_train = to_categorical(y_train) # Transforme le label en sortie en vecteur

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1/3, random_state=0)
###Creation du model
####################
model = Sequential()
#Ajout des couches de neurones
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compilation du model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)

y_validation = model.predict(X_validation)
y_validation = np.around(y_validation,0)
y_validation = y_validation.astype(int)


Image_id = [ i for i in range(1,n_test+1)]
label_prediction = np.zeros(n_test,dtype=int)
for i in range (n_test) :
    for j in range(np.shape(y_validation[:][i])[0]):
        if (y_validation[i][j]==1):
            label_prediction[i]=j
#print(label_prediction)
print(np.shape(label_prediction))
print(np.shape(Image_id))
output = pd.DataFrame({'ImageId': Image_id, 'Label': label_prediction})
#print(output)
output.to_csv('my_submission_digit_test.csv', index=False)

"""for i in range(10,18):
    img = train_data.loc[i,:].values
    img_resiezd = np.resize(img,(28,28))
    plt.imshow(img_resiezd)
    plt.show()"""


