"""
Investigate which characters are causing test case failures.
"""
import pickle
import inflect
import numpy as np

# Load the character encoder to see the training vocabulary
with open('saved_models/character_label_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

training_chars = set(encoder.classes_)

print("="*80)
print("TRAINING CHARACTER VOCABULARY")
print("="*80)
print(f"Total characters in training vocab: {len(training_chars)}")
print(f"Characters: {sorted(training_chars)}")
print()

# Check number words 50-60
p = inflect.engine()
print("="*80)
print("CHECKING NUMBER WORDS 50-60")
print("="*80)
for num in range(50, 61):
    num_word = p.number_to_words(num)
    chars_in_num = set(num_word)
    unseen_chars = chars_in_num - training_chars
    if unseen_chars:
        print(f"{num} -> '{num_word}': UNSEEN CHARS: {unseen_chars}")
    else:
        print(f"{num} -> '{num_word}': ✓ all chars seen")
print()

# Check negative number words
print("="*80)
print("CHECKING NEGATIVE NUMBER WORDS")
print("="*80)
for num in range(-10, 0):
    num_word = f"negative {p.number_to_words(abs(num))}"
    chars_in_num = set(num_word)
    unseen_chars = chars_in_num - training_chars
    if unseen_chars:
        print(f"{num} -> '{num_word}': UNSEEN CHARS: {unseen_chars}")
    else:
        print(f"{num} -> '{num_word}': ✓ all chars seen")
print()

# Check openings
from evaluate_generalization import (
    OPENINGS_IN_DISTRIBUTION,
    OPENINGS_NOVEL_BUT_SIMILAR,
    OPENINGS_VERY_DIFFERENT
)

print("="*80)
print("CHECKING OPENINGS - IN-DISTRIBUTION")
print("="*80)
for opening in OPENINGS_IN_DISTRIBUTION:
    chars_in_opening = set(opening)
    unseen_chars = chars_in_opening - training_chars
    if unseen_chars:
        print(f"'{opening}': UNSEEN CHARS: {unseen_chars}")
    else:
        print(f"'{opening}': ✓ all chars seen")
print()

print("="*80)
print("CHECKING OPENINGS - NOVEL BUT SIMILAR")
print("="*80)
for opening in OPENINGS_NOVEL_BUT_SIMILAR:
    chars_in_opening = set(opening)
    unseen_chars = chars_in_opening - training_chars
    if unseen_chars:
        print(f"'{opening}': UNSEEN CHARS: {unseen_chars}")
    else:
        print(f"'{opening}': ✓ all chars seen")
print()

print("="*80)
print("CHECKING OPENINGS - VERY DIFFERENT")
print("="*80)
for opening in OPENINGS_VERY_DIFFERENT:
    chars_in_opening = set(opening)
    unseen_chars = chars_in_opening - training_chars
    if unseen_chars:
        print(f"'{opening}': UNSEEN CHARS: {unseen_chars}")
    else:
        print(f"'{opening}': ✓ all chars seen")
