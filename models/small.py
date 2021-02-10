from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, BatchNormalization
from tensorflow.keras.models import Sequential


def get_model(settings, n_notes):
    # Initializing the classifier Network
    classifier = Sequential([
        Conv1D(16, kernel_size=2, padding="causal", activation="relu"),
        BatchNormalization(),

        LSTM(32, input_shape=(settings.window_size, n_notes), return_sequences=True, dropout=0.2),
        LSTM(32, dropout=0.1),

        Dense(16, activation='relu'),

        Dense(n_notes, activation=settings.final_activation),
    ])

    # Compiling the network
    classifier.compile(loss=settings.loss,
                       optimizer=settings.optimizer,
                       metrics=['accuracy'])

    return classifier
