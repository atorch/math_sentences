import inflect


INFLECT_ENGINE = inflect.engine()

# Note: this list contains "zero", "one", "two", "three", etc
NUMBERS = [INFLECT_ENGINE.number_to_words(x) for x in range(40)]

OPENINGS = [
    "",
    "Clearly ",
    "Did you know that ",
    "Hey ",
    "Hey?! ",
    "I need to know ",
    "I think ",
    "So, clearly ",
    "So... ",
    "Um, barf ",
    "Well, tell me, ",
    "What do you think ",
    "What's ",
    "You're saying that ",
]
EQUALITIES = ["equals", "is", "="]
OPERATIONS = ["plus", "+", "added to", "and"]

CHARACTERS = list(
    set([c for c in "".join(OPENINGS + NUMBERS + OPERATIONS + EQUALITIES)])
)
