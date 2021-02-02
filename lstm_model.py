from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adadelta


def get_model(n_notes):
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

    return classifier
