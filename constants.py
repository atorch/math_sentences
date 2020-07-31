import inflect


INFLECT_ENGINE = inflect.engine()

# Note: this list contains "zero", "one", "two", "three", etc
NUMBERS = [INFLECT_ENGINE.number_to_words(x) for x in range(50)]

OPENINGS = [
    "",
    "And what is ",
    "Clearly ",
    "Did you know that ",
    "Eighteen is a number, but ignore it. What's ",
    "Eleven is a number -- ignore it. What's ",
    "Hey ",
    "Hey?! ",
    "I need to know ",
    "I think ",
    "Ignore the number nine. What's ",
    "Ignore this text. Answer this: ",
    "My question is: what's ",
    "Nine is a number -- ignore it. What is ",
    "Now, what is ",
    "One question I have is: ",
    "One thing I thought of is ",
    "One two three are numbers. What is ",
    "Question three: ",
    "Question: ",
    "Seventeen is a number. What is ",
    "So what is ",
    "So, clearly ",
    "So... ",
    "So: what is ",
    "Therefore ",
    "Um, barf! I hate math... ",
    "Well, tell me, ",
    "What do you think ",
    "What is ",
    "What's ",
    "You're saying that ",
    "You've seen that ",
]

EQUALITIES = ["="]

OPERATIONS_PLUS = ["plus", "+", "added to", "and"]
OPERATIONS_MINUS = ["minus", "-"]

CHARACTERS = list(
    set([c for c in "".join(OPENINGS + NUMBERS + OPERATIONS_MINUS + OPERATIONS_PLUS + EQUALITIES)])
)
