import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import losses, optimizers
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
    OPENINGS,
    OPERATIONS_MINUS,
    OPERATIONS_PLUS,
)


def simulate_sentence():

    # Note: w2n requires str inputs (not numpy strings)
    number1 = str(np.random.choice(NUMBERS))
    number2 = str(np.random.choice(NUMBERS))

    equals = np.random.choice(EQUALITIES)
    opening = np.random.choice(OPENINGS)

    if np.random.uniform() < 0.50:
        operation = np.random.choice(OPERATIONS_PLUS)
        result = w2n.word_to_num(number1) + w2n.word_to_num(number2)

    else:
        operation = np.random.choice(OPERATIONS_MINUS)
        result = w2n.word_to_num(number1) - w2n.word_to_num(number2)

    # Note: the opening is independent of the result,
    #  so the model should learn to ignore it
    sentence = f"{opening}{number1} {operation} {number2} {equals}"

    # Note: this function can only return a finite number of possible sentences.
    #  Eventually, the model will see all of them...

    # TODO Different distribution for train and validation generators, so that we
    #  can see how the model does on unseen sentences similar to those in the training set

    # TODO Could fit a multi-objective model that tries to predict the result _and_
    #  something else, e.g. the second number in the operation
    return sentence, result


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


def get_generator(character_label_encoder, batch_size=20):

    n_character_classes = len(character_label_encoder.classes_)

    while True:

        # Note: the sentences in each batch have to be of the same length
        n_characters_in_sentence = np.random.choice(np.arange(30, 65))

        batch_X_shape = (batch_size, n_characters_in_sentence, n_character_classes)
        batch_X = np.zeros(batch_X_shape)

        batch_Y = np.zeros((batch_size,))

        for idx in range(batch_size):

            sentence, result = simulate_sentence()

            sentence = force_sentence_to_n_chars(sentence, n_characters_in_sentence)

            characters_one_hot = get_characters_one_hot_encoded(
                sentence, character_label_encoder
            )

            batch_X[idx] = characters_one_hot
            batch_Y[idx] = result

        # Note: the generator returns tuples of (inputs, targets)
        yield (batch_X, batch_Y)


def get_model(character_label_encoder, dropout=0.1):

    n_character_classes = len(character_label_encoder.classes_)

    # Note: input shape None-by-n_character_classes allows for arbitrary length sentences
    input_layer = Input(shape=(None, n_character_classes))

    gru1 = GRU(units=32, return_sequences=True, dropout=dropout,)(input_layer)

    # Note: activation=None means linear activation
    gru2 = GRU(units=1, activation=None,)(gru1)

    model = Model(inputs=input_layer, outputs=gru2)

    nadam = optimizers.Nadam()

    model.compile(
        optimizer=nadam, loss=losses.mean_squared_error, metrics=["mean_squared_error"],
    )

    print(model.summary())

    return model


def print_prediction(model, character_label_encoder, sentence, result):

    characters_one_hot = get_characters_one_hot_encoded(
        sentence, character_label_encoder
    )
    characters_one_hot = np.expand_dims(characters_one_hot, axis=0)

    prediction = np.round(model.predict(tf.convert_to_tensor(characters_one_hot))[0][0], 2)
    print(
        f"Input sentence '{sentence}', correct result {result}, prediction {prediction}"
    )


def main(epochs=500):

    character_label_encoder = LabelEncoder().fit(CHARACTERS)

    training_generator = get_generator(character_label_encoder)

    model = get_model(character_label_encoder)

    history = model.fit(
        x=training_generator, steps_per_epoch=100, epochs=epochs, verbose=True,
    )

    # First, let's see how the model does on sentences that it may have seen during training
    for idx in range(5):

        sentence, result = simulate_sentence()
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
        len(OPENINGS) * (len(NUMBERS) ** 2) * len(OPERATIONS_MINUS + OPERATIONS_PLUS) * len(EQUALITIES)
    )

    print(
        f"Note: the maximum number of possible (unique) sentences generated by the training generator is {n_possible_sentences}"
    )


if __name__ == "__main__":
    main()
