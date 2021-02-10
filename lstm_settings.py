from tf import optimizers
from collections import namedtuple

TrainSettings = namedtuple('Encoded', 'ticks_per_second, window_size threshold_scale '
                                      'composer network loss optimizer final_activation')


def get_model_id(settings, n_notes):
    return f"{settings.composer}_" \
           f"{settings.network}_" \
           f"{settings.loss}_" \
           f"{settings.optimizer._name}_" \
           f"{settings.final_activation}_" \
           f"n{n_notes}_" \
           f"tps{settings.ticks_per_second}_" \
           f"ws{settings.window_size}"


base_settings = TrainSettings(
    ticks_per_second=8,
    window_size=96,
    threshold_scale=.7,

    # composer=None,
    composer='bach',

    # network='small',
    # network='medium',
    network='big',

    loss='categorical_crossentropy',
    # loss='binary_crossentropy',
    # loss='mse',

    # optimizer = SGD(learning_rate=0.01, momentum=0.7, nesterov=True),
    optimizer=optimizers.Nadam(learning_rate=0.005),

    # final_activation='softmax',
    final_activation='sigmoid',
)
