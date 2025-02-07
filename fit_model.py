import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import callbacks, losses, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Dropout,
    Input,
    MultiHeadAttention,
    LayerNormalization,
    Dense,
    Add,
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


# Transformer block
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = Add()([inputs, x])
    
    x = Dense(ff_dim, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    x = Add()([res, x])
    return LayerNormalization(epsilon=1e-6)(x)


def get_model(character_label_encoder, head_size=256, num_heads=4, ff_dim=256, dropout=0.1, num_transformer_blocks=4):

    n_character_classes = len(character_label_encoder.classes_)

    input_layer = Input(shape=(None, n_character_classes))

    x = input_layer
    for _ in range(num_transformer_blocks):
        x = transformer_block(x, head_size, num_heads, ff_dim, dropout)

    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(n_character_classes)(x)

    # Output layers for regression tasks
    output_result = Dense(1, activation=None, name=OUTPUT_RESULT)(x)
    output_first_number = Dense(1, activation=None, name=OUTPUT_FIRST_NUMBER)(x)

    model = Model(inputs=input_layer, outputs=[output_result, output_first_number])

    nadam = optimizers.Nadam()

    model.compile(
        optimizer=nadam, loss=losses.mean_squared_error, metrics=["mean_squared_error"],
    )

    print(model.summary())

    return model

# The rest of the code remains unchanged (simulate_sentence, get_characters_one_hot_encoded, force_sentence_to_n_chars, get_generator, get_output_names, print_prediction, main)


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


if __name__ == "__main__":
    main()
