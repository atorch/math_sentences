import inflect
import numpy as np
from sklearn.preprocessing import LabelEncoder
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


INFLECT_ENGINE = inflect.engine()

# Note: this list contains "zero", "one", "two", "three", etc
NUMBERS = [INFLECT_ENGINE.number_to_words(x) for x in range(25)]

OPENINGS = ["", "Did you know that ", "I think ", "What do you think "]
EQUALITIES = ["equals", "is", "="]
OPERATIONS = ["plus", "+", "added to"]

CHARACTERS = list(
    set([c for c in "".join(OPENINGS + NUMBERS + OPERATIONS + EQUALITIES)])
)


def simulate_sentence():

    # TODO Possiblity of adding three numbers?
    # Note: w2n requires str inputs (not numpy strings)
    number1 = str(np.random.choice(NUMBERS))
    number2 = str(np.random.choice(NUMBERS))

    equals = np.random.choice(EQUALITIES)
    opening = np.random.choice(OPENINGS)
    operation = np.random.choice(OPERATIONS)

    # Note: the opening is independent of the result,
    #  so the model should learn to ignore it
    sentence = f"{opening}{number1} {operation} {number2} {equals}"

    result = w2n.word_to_num(number1) + w2n.word_to_num(number2)

    # Note: this function can only return a finite number of possible sentences.
    #  Eventually, the model will see all of them...
    return sentence, result


def get_characters_one_hot_encoded(sentence, character_label_encoder):

    characters = [c for c in sentence]
    characters_encoded = character_label_encoder.transform(characters)

    n_character_classes = len(character_label_encoder.classes_)

    # TODO Could use an embedding layer at the beginning of the model instead of doing this
    return to_categorical(characters_encoded, num_classes=n_character_classes)


def get_generator(character_label_encoder, batch_size=20):

    n_character_classes = len(character_label_encoder.classes_)

    # TODO Hack
    n_characters_in_sentence = 30

    while True:

        # Note: the sentences in each batch have to be of the same length
        batch_X_shape = (batch_size, n_characters_in_sentence, n_character_classes)
        batch_X = np.zeros(batch_X_shape)

        batch_Y = np.zeros((batch_size,))

        for idx in range(batch_size):

            sentence, result = simulate_sentence()

            # TODO Hack to force sentence to be of length n_characters_in_sentence
            if len(sentence) > n_characters_in_sentence:
                sentence = sentence[:n_characters_in_sentence]

            elif len(sentence) < n_characters_in_sentence:
                n_characters_missing = n_characters_in_sentence - len(sentence)
                whitespace = "".join([" "] * n_characters_missing)
                sentence = f"{whitespace}{sentence}"

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


def main():

    character_label_encoder = LabelEncoder().fit(CHARACTERS)

    training_generator = get_generator(character_label_encoder)

    model = get_model(character_label_encoder)

    history = model.fit(
        x=training_generator, steps_per_epoch=100, epochs=80, verbose=True,
    )

    for idx in range(3):

        sentence, result = simulate_sentence()

        characters_one_hot = get_characters_one_hot_encoded(
            sentence, character_label_encoder
        )
        characters_one_hot = np.expand_dims(characters_one_hot, axis=0)

        prediction = model.predict(characters_one_hot)[0][0]
        print(
            f"Input sentence '{sentence}', correct result {result}, prediction {prediction}"
        )


if __name__ == "__main__":
    main()
