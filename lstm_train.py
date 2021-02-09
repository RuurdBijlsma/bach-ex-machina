import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])

from matplotlib import pyplot as plt
from models import get_model
from prepare_data import get_processed_data, ts_generator
from lstm_settings import settings, get_model_id
from lstm_gen import generate


def main():
    # tf.debugging.set_log_device_placement(True)
    # Importing the output

    (train, test, validation), _ = get_processed_data(settings.ticks_per_second, settings.composer)

    train[train > 0] = 1
    test[test > 0] = 1
    validation[validation > 0] = 1

    train = ts_generator(train, settings.window_size)
    test = ts_generator(test, settings.window_size)
    validation = ts_generator(validation, settings.window_size)

    lstm_train(settings, train, validation)

    lstm_test(settings, test)

    generate(settings)


def lstm_train(settings, train, validation):
    n_notes = train.data.shape[1]
    print(f"Training model: {get_model_id(settings, n_notes)}")

    # Initializing the classifier Network
    classifier = get_model(settings, n_notes)

    checkpoint_path = f"output/{get_model_id(settings, n_notes)}"

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True,
                                                     save_weights_only=False, verbose=1)

    # Fitting the output to the model
    try:
        history = classifier.fit(train,
                                 epochs=25,
                                 callbacks=[cp_callback],
                                 validation_data=validation)
    except KeyboardInterrupt:
        history = None
        print('\nIntercepted KeyboardInterrupt, evaluating model.')
    print(classifier.summary())

    if history is not None:
        # Plot accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f"output/acc_{get_model_id(settings, n_notes)}.png")
        plt.show()

        # Plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f"output/loss_{get_model_id(settings, n_notes)}.png")
        plt.show()


def lstm_test(settings, test):
    n_notes = test.data.shape[1]
    print(f"Testing model from checkpoint: {get_model_id(settings, n_notes)}")

    checkpoint_path = f"output/{get_model_id(settings, n_notes)}"
    classifier = tf.keras.models.load_model(checkpoint_path)

    test_loss, test_acc = classifier.evaluate(test)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    log = 'output/results.csv'
    txt = []
    if not os.path.isfile(log):
        txt.append('composer,network,loss,optimizer,activation,acc,loss')

    txt.append(
        f'{settings.composer},{settings.network},{settings.loss},{settings.optimizer._name},{settings.final_activation},{test_loss},{test_acc}')

    with open(log, 'a') as f:
        f.writelines('\n'.join(txt) + '\n')


if __name__ == '__main__':
    main()
