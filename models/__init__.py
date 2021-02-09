import importlib


def get_model(settings, n_notes):
    model_module = importlib.import_module(f"models.{settings.network}")
    return model_module.get_model(settings, n_notes)
