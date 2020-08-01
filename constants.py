import inflect


INFLECT_ENGINE = inflect.engine()

# Note: this list contains "zero", "one", "two", "three", etc
NUMBERS = [INFLECT_ENGINE.number_to_words(x) for x in range(50)]

OPENINGS_TRAINING = [
    "",
    "And what is ",
    "Because seven eight nine. ",
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
    "Zero eight something twelve... ",
]

OPENINGS_VALIDATION = [
    "Because ten eleven twelve :-) ",
    "Beep boop beep boop, I am a robot. ",
    "Five nine zero pandas. ",
    "One and three are numbers, but ignore them. ",
    "So, what is ",
    "The quick brown fox jumps over the lazy dog... ",
    "Validation question: what is ",
    "Why is six afraid of seven ? ",
    "You've never seen this string. Can you ignore it? ",
]

EQUALITIES = ["="]

OPERATIONS_PLUS = ["plus", "+", "added to", "and"]
OPERATIONS_MINUS = ["minus", "-"]

CHARACTERS = list(
    set([c for c in "".join(OPENINGS_VALIDATION + OPENINGS_TRAINING + NUMBERS + OPERATIONS_MINUS + OPERATIONS_PLUS + EQUALITIES)])
)

OUTPUT_RESULT = "result"
OUTPUT_FIRST_NUMBER = "first_num"
