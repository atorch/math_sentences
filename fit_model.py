import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import callbacks, losses, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dropout,
    GRU,
    Input,
    LSTM,
)
from word2number import w2n

from constants import (
    CHARACTERS,
    EQUALITIES,
    NUMBERS,
    OPENINGS_TRAINING,
    OPENINGS_VALIDATION,
    OPERATIONS_MINUS,
    OPERATIONS_PLUS,
    OUTPUT_RESULT,
    OUTPUT_FIRST_NUMBER,
)


def simulate_sentence(openings):

    # Note: w2n requires str inputs (not numpy strings)
    number1_as_str = str(np.random.choice(NUMBERS))
    number2_as_str = str(np.random.choice(NUMBERS))

    number1_as_float = w2n.word_to_num(number1_as_str)
    number2_as_float = w2n.word_to_num(number2_as_str)

    equals = np.random.choice(EQUALITIES)
    opening = np.random.choice(openings)

    if np.random.uniform() < 0.50:
        operation = np.random.choice(OPERATIONS_PLUS)
        result = number1_as_float + number2_as_float

    else:
        operation = np.random.choice(OPERATIONS_MINUS)
        result = number1_as_float - number2_as_float

    # Note: the opening is independent of the result,
    #  so the model should learn to ignore it
    sentence = f"{opening}{number1_as_str} {operation} {number2_as_str} {equals}"

    # Note: this function can only return a finite number of possible sentences.
    #  Eventually, the model will see all of them...

    # TODO Different distribution for train and validation generators, so that we
    #  can see how the model does on unseen sentences similar to those in the training set

    # TODO Could fit a multi-objective model that tries to predict the result _and_
    #  something else, e.g. the second number in the operation
    return sentence, result, number1_as_float


def get_characters_one_hot_encoded(sentence, character_label_encoder):

    characters = [c for c in sentence]
    characters_encoded = character_label_encoder.transform(characters)

    n_character_classes = len(character_label_encoder.classes_)

    # TODO Could use an embedding layer at the beginning of the model instead of doing this
    return to_categorical(characters_encoded, num_classes=n_character_classes)


def force_sentence_to_n_chars(sentence, n_characters_in_sentence):

    if len(sentence) > n_characters_in_sentence:

        # Note: we use the _end_ of the sentence, which is the part that contains math
        #  (as opposed to the sentence opening, which might contain non-math words)
        sentence = sentence[-n_characters_in_sentence:]

    elif len(sentence) < n_characters_in_sentence:
        n_characters_missing = n_characters_in_sentence - len(sentence)
        # TODO Would it help to add random characters instead of whitespace?
        whitespace = "".join([" "] * n_characters_missing)
        sentence = f"{whitespace}{sentence}"

    return sentence


def get_generator(character_label_encoder, openings, batch_size=20):

    n_character_classes = len(character_label_encoder.classes_)

    while True:

        # Note: the sentences in each batch have to be of the same length
        n_characters_in_sentence = np.random.choice(np.arange(30, 65))

        batch_X_shape = (batch_size, n_characters_in_sentence, n_character_classes)
        batch_X = np.zeros(batch_X_shape)

        # Note: the model has multiple objectives/outputs
        batch_result = np.zeros((batch_size,))
        batch_first_number = np.zeros((batch_size,))

        for idx in range(batch_size):

            sentence, result, first_number = simulate_sentence(openings)

            sentence = force_sentence_to_n_chars(sentence, n_characters_in_sentence)

            characters_one_hot = get_characters_one_hot_encoded(
                sentence, character_label_encoder
            )

            batch_X[idx] = characters_one_hot
            batch_result[idx] = result
            batch_first_number[idx] = first_number

        targets = {OUTPUT_FIRST_NUMBER: batch_first_number, OUTPUT_RESULT: batch_result}

        # Note: the generator returns tuples of (inputs, targets)
        yield (batch_X, targets)


def get_model(character_label_encoder, dropout=0.05, n_units=64):

    n_character_classes = len(character_label_encoder.classes_)

    # Note: input shape None-by-n_character_classes allows for arbitrary length sentences
    input_layer = Input(shape=(None, n_character_classes))

    gru1 = GRU(units=n_units, return_sequences=True, dropout=dropout,)(input_layer)
    gru2 = GRU(units=n_units, return_sequences=True, dropout=dropout,)(gru1)

    # Note: activation=None means linear activation (used for regression output)
    gru_result = GRU(units=1, activation=None, name=OUTPUT_RESULT)(gru2)
    gru_first_number = GRU(units=1, activation=None, name=OUTPUT_FIRST_NUMBER)(gru2)

    model = Model(inputs=input_layer, outputs=[gru_result, gru_first_number])

    nadam = optimizers.Nadam()

    model.compile(
        optimizer=nadam, loss=losses.mean_squared_error, metrics=["mean_squared_error"],
    )

    print(model.summary())

    return model


def get_output_names(model):

    # Note: example name is TODO -- get the part before the "/"
    return [x.op.name.split("/")[0] for x in model.outputs]


def print_prediction(model, character_label_encoder, sentence, result):

    output_names = get_output_names(model)
    result_index = np.where([o.startswith(OUTPUT_RESULT) for o in output_names])[0][0]

    characters_one_hot = get_characters_one_hot_encoded(
        sentence, character_label_encoder
    )
    characters_one_hot = np.expand_dims(characters_one_hot, axis=0)

    prediction = model.predict(tf.convert_to_tensor(characters_one_hot))[result_index][
        0
    ][0]
    print(
        f"Input sentence '{sentence}', correct result {result}, prediction {prediction:.2f}"
    )


def main(max_epochs=500):

    character_label_encoder = LabelEncoder().fit(CHARACTERS)

    training_generator = get_generator(
        character_label_encoder, openings=OPENINGS_TRAINING
    )
    validation_generator = get_generator(
        character_label_encoder, openings=OPENINGS_VALIDATION
    )

    model = get_model(character_label_encoder)

    history = model.fit(
        x=training_generator,
        steps_per_epoch=100,
        epochs=max_epochs,
        verbose=True,
        callbacks=[
            callbacks.EarlyStopping(
                patience=20, monitor="val_loss", restore_best_weights=True, verbose=True
            )
        ],
        validation_data=validation_generator,
        validation_steps=50,
    )

    # First, let's see how the model does on sentences that it may have seen during training
    for idx in range(5):

        sentence, result, number1 = simulate_sentence(openings=OPENINGS_TRAINING)
        print_prediction(model, character_label_encoder, sentence, result)

    # Next, let's start with a version of "one + one" that is part of the training set, and
    #  gradually modify it so that we are extrapolating (i.e. predicting on an unseen sentence)
    result = 2
    for sentence in [
        "What is one + one =",
        "Can you answer one + one =",
        "Never seen before: what is one + one =",
        "What is one + one?",
        "Sentence you've never seen before: what is one + one??",
    ]:
        print_prediction(model, character_label_encoder, sentence, result)

    # Now let's see how the model does when extrapolating to an unseen sentence
    #  that is completely different from those in the training set (the model has never seen negative input numbers!)
    unseen_sentence = "This is a sentence you've never seen before. What is negative one plus negative two?"
    unseen_result = -3
    print_prediction(model, character_label_encoder, unseen_sentence, unseen_result)

    # See the sentence construction logic in simulate_sentence
    n_possible_sentences = (
        len(OPENINGS_TRAINING)
        * (len(NUMBERS) ** 2)
        * len(OPERATIONS_MINUS + OPERATIONS_PLUS)
        * len(EQUALITIES)
    )

    print(
        f"Note: the maximum number of possible (unique) sentences generated by the training generator is {n_possible_sentences}"
    )

    # Save the model and encoder
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)

    model_dir = Path("saved_models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / "math_model.keras"
    encoder_path = model_dir / "character_label_encoder.pkl"

    # Save Keras model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save label encoder using pickle
    with open(encoder_path, 'wb') as f:
        pickle.dump(character_label_encoder, f)
    print(f"Character encoder saved to {encoder_path}")

    # Run systematic generalization evaluation
    print("\n" + "="*80)
    print("RUNNING GENERALIZATION EVALUATION")
    print("="*80)

    try:
        from evaluate_generalization import run_evaluation
        run_evaluation(model, character_label_encoder)
    except ImportError as e:
        print(f"Could not import evaluate_generalization: {e}")
        print("Skipping generalization evaluation.")

    return model, character_label_encoder


if __name__ == "__main__":
    main()
