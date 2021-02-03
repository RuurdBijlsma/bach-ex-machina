import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from lstm_model import get_model
from prepare_data import get_processed_data


def main():
    # Importing the data
    composer = "bach"
    compress = 1

    (train, test, validation), (start, end) = get_processed_data(composer, compress)
    n_notes = train.shape[1]
    checkpoint_path = f"data/{composer}_checkpoint_n{n_notes}_c{compress}.ckpt"

    train[train > 0] = 1
    test[test > 0] = 1

    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())
    # tf.debugging.set_log_device_placement(True)

    validation[validation > 0] = 1

    # (train_x, train_y) = to_input_output(train)
    # (test_x, test_y) = to_input_output(test)
    # (val_x, val_y) = to_input_output(validation)

    # test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[1]))
    # train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[1]))
    # val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[1]))

    window_size = 30
    test = tf.keras.preprocessing.sequence.TimeseriesGenerator(test, test, window_size)
    train = tf.keras.preprocessing.sequence.TimeseriesGenerator(train, train, window_size)
    validation = tf.keras.preprocessing.sequence.TimeseriesGenerator(validation, validation, window_size)

    # Initializing the classifier Network
    classifier = get_model(n_notes, window_size)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                                                     save_weights_only=True, verbose=1)

    # Fitting the data to the model
    classifier.fit(train,
                   epochs=15,
                   callbacks=[cp_callback],
                   validation_data=validation)
    print(classifier.summary())

    test_loss, test_acc = classifier.evaluate(test)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    main()
