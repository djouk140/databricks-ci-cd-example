
from notebooks.train_model import train_and_evaluate


def test_model_training():

    model, accuracy, input_example = train_and_evaluate()

    # Verifier que le modele existe
    assert model is not None

    # Verifier la precision
    assert 0 <= accuracy <= 1

    # Verifier les predictions
    assert len(model.predict(input_example)) == 5


def test_model_accuracy():

    model, accuracy, input_example = train_and_evaluate()

    # Le modele doit atteindre une precision minimale
    assert accuracy >= 0.80
