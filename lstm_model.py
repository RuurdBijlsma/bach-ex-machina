from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta


def get_model(n_notes, window_size):
    # Initializing the classifier Network
    classifier = Sequential([
        LSTM(128, input_shape=(window_size, n_notes), return_sequences=True),
        Dropout(0.2),

        LSTM(128),

        Dense(64, activation='relu'),
        Dropout(0.2),

        Dense(n_notes, activation='sigmoid'),
    ])

    # Compiling the network
    classifier.compile(loss='binary_crossentropy',
                       optimizer=Adadelta(lr=0.001, decay=1e-6),
                       metrics=['accuracy'])

    return classifier
