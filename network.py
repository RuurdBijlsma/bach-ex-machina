import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam, Adadelta
import numpy as np
from prepare_data import get_processed_data, to_input_output


def main():
    # Importing the data
    composer = "bach"
    (train, test, validation), (start, end) = get_processed_data(composer)
    n_notes = train.shape[1]

    train[train > 0] = 1
    test[test > 0] = 1
    validation[validation > 0] = 1

    (train_x, train_y) = to_input_output(train)
    (test_x, test_y) = to_input_output(test)
    (val_x, val_y) = to_input_output(validation)

    # test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1]))
    # train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))
    # val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1]))

    # Initializing the classifier Network
    classifier = Sequential()

    # Adding the input LSTM network layer
    classifier.add(LSTM(128, input_shape=(n_notes, 1), return_sequences=True))
    classifier.add(Dropout(0.2))

    # Adding a second LSTM network layer
    classifier.add(LSTM(128))

    # Adding a dense hidden layer
    classifier.add(Dense(64, activation='relu'))
    classifier.add(Dropout(0.2))

    # Adding the output layer
    classifier.add(Dense(n_notes, activation='sigmoid'))

    # Compiling the network
    classifier.compile(loss='binary_crossentropy',
                       optimizer=Adadelta(lr=0.001, decay=1e-6),
                       metrics=['accuracy'])

    # Fitting the data to the model
    classifier.fit(train_x,
                   train_y,
                   epochs=3,
                   validation_data=(val_x, val_y))
    print(classifier.summary())

    test_loss, test_acc = classifier.evaluate(test_x, test_y)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    main()
