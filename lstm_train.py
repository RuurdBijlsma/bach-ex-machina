import tensorflow as tf
import settings
from lstm_model import get_model
from prepare_data import get_processed_data, ts_generator


def main():
    # Importing the data
    # composer = "bach"
    composer = None
    tf.debugging.set_log_device_placement(True)

    (train, test, validation), _ = get_processed_data(composer)
    n_notes = train.shape[1]

    train[train > 0] = 1
    test[test > 0] = 1
    validation[validation > 0] = 1

    window_size = 30
    train = ts_generator(train, window_size)
    test = ts_generator(test, window_size)
    validation = ts_generator(validation, window_size)

    # Initializing the classifier Network
    classifier = get_model(n_notes, window_size)

    checkpoint_path = f"data/{composer}_checkpoint_n{n_notes}_tps{settings.ticks_per_second}"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                                                     save_weights_only=False, verbose=1)

    # Fitting the data to the model
    classifier.fit(train,
                   epochs=20,
                   callbacks=[cp_callback],
                   validation_data=validation)
    print(classifier.summary())

    test_loss, test_acc = classifier.evaluate(test)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))


if __name__ == '__main__':
    main()
