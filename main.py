import tensorflow as tf
import keras

from typing import List

from string import punctuation
from os import listdir
from numpy import array
import numpy as np
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from keras_preprocessing.sequence import pad_sequences



def main():
    xtrain = [] # 2D list, list of feature vectors
    ytrain = [] # list of labels

    xtest = []
    ytest = []

    with open('Sentiment140_Vector_Representation_5pct.csv') as file:
        lines = file.readlines()

        count = 0
        for line in lines:
            # if count > 100:
            #     break
            count += 1
            label = int(line[0])
            label = 1 if label == 4 else 0
            parts = line.split('"')

            # [0.04537600320246485, 0.19668344408273697, ... ]
            data: str = parts[1]

            data = data[1:-1]
            data.replace(' ', '')

            nums: List[str] = data.split(',')
            feature_vector: List[float] = [float(x) for x in nums]

            xtrain.append(np.asarray(feature_vector))
            ytrain.append(np.asarray([label]))
            
            if count % 5 == 0:
                xtest.append(np.asarray(feature_vector))
                ytest.append(np.asarray([label]))
            else:
                xtrain.append(np.asarray(feature_vector))
                ytrain.append(np.asarray([label]))
        
    size = len(xtrain)
    train_size = int(size * .8)
    #print(xtrain)
    #xtrain = pad_sequences(np.asarray(xtrain), maxlen=200, padding='post')
    #xtest = pad_sequences(np.asarray(xtest), maxlen=200, padding='post')
    # xtrain = tf.data.Dataset.from_tensor_slices(xtrain)
    #print(xtrain)
    #print(ytrain)
    #print()
    xtrain = np.asarray(xtrain)
    xtest = np.asarray(xtest)
    ytrain = np.array(ytrain)
    ytest = np.array(ytest)
    print(xtrain)
    print(ytrain)

    # xtrain, xtest = keras.utils.split_dataset(xtrain, left_size=train_size, shuffle=True)
    # ytrain, ytest = keras.utils.split_dataset(ytrain, left_size=train_size, shuffle=True)

    run_model(xtrain, ytrain, xtest, ytest)


def run_model(xtrain, ytrain, xtest, ytest):
    print('shapes')
    print(xtrain.shape)
    print(ytrain.shape)
    print('building model')
    # define model
    model = Sequential()
    model.add(tf.keras.layers.Embedding(200, 64, input_length=200))
    model.add(Conv1D(filters=100, kernel_size=8, activation='relu'))
    # model.add(Conv1D(filters=100, kernel_size=8, activation='relu', input_shape=(None,200)))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    print('fitting model')
    print('types')
    print(type(xtrain))
    print(type(ytrain))
    print(model.summary())
    model.fit(xtrain, ytrain, epochs=10, verbose=2)
    # evaluate
    print('evaluating model')
    loss, acc = model.evaluate(xtest, ytest, verbose=0)
    print('Test Accuracy: %f' % (acc*100))

    print(model.summary())

if __name__ == '__main__':
    main()