# math_sentences

A character-by-character RNN playground for exploring how neural networks learn (or memorize) basic arithmetic from natural language (inspired by [this example](https://youtu.be/0VH1Lim8gL8?t=1417)).

## How It Works

This project demonstrates that a character-level RNN can learn to extract and solve (or memorize?) arithmetic problems from natural language sentences, while ignoring irrelevant surrounding text.

### Architecture

- **Input**: Variable-length sentences (30-65 characters) encoded character-by-character as one-hot vectors
- **Model**:
  - 2 stacked GRU layers (64 units each) with dropout
  - 2 separate output heads (both single-unit GRUs with linear activation)
- **Outputs** (multi-objective regression):
  1. `result`: The answer to the math problem
  2. `first_num`: The first number in the equation (auxiliary task to help learning)

### Training Data Generation

The model trains on synthetically generated sentences with the pattern:
```
{opening}{number1} {operation} {number2} {equals}
```

- **Numbers**: "zero" through "forty-nine" (50 values, 0-49)
- **Operations**: Plus variants ("plus", "+", "added to", "and") and minus variants ("minus", "-")
- **Openings**: 35 training phrases and 9 validation phrases (e.g., "What is ", "Hey?! ", "Ignore this text. Answer this: ")
- **Total possible unique sentences**: 525,000

The key insight: by varying the openings (including distractors like "Because seven eight nine. " and "Um, barf! I hate math... "), the model learns to ignore irrelevant text and focus on the mathematical operation.

### Training Process

- Batch size: 20
- Steps per epoch: 100
- Early stopping: patience=20, monitoring validation loss
- Variable sentence lengths within each batch forces the model to handle different input sizes
- Separate validation openings test generalization to unseen sentence structures

## Usage

### Training a New Model

```bash
sudo docker build ~/math_sentences --tag=math_docker
sudo docker run -it -v ~/math_sentences:/home/math_sentences math_docker bash
python fit_model.py
```

This will:
1. Train the model with early stopping
2. Save the model to `saved_models/math_model.keras`
3. Save the character encoder to `saved_models/character_label_encoder.pkl`
4. Run generalization evaluation (see below)

### Evaluating a Saved Model

To re-run evaluation on a previously trained model without retraining:

```bash
python load_and_evaluate.py
```

This loads the saved model and runs the full generalization analysis.

## Current Results

### Predictions on training generator sentences:

```bash
Input sentence 'My question is: what's fourteen minus fourteen =', correct result 0, prediction -0.5
Input sentence 'Hey forty-one plus eighteen =', correct result 59, prediction 59.06999969482422
Input sentence 'Um, barf! I hate math... forty-two - forty-nine =', correct result -7, prediction -6.510000228881836
```

The model performs well on sentences from the training distribution.

### Extrapolation (out-of-distribution)

```bash
Input sentence 'What is one + one =', correct result 2, prediction 3.9800000190734863
Input sentence 'Can you answer one + one =', correct result 2, prediction 5.880000114440918
Input sentence 'Never seen before: what is one + one =', correct result 2, prediction 3.9600000381469727
Input sentence 'What is one + one?', correct result 2, prediction 2.009999990463257
Input sentence 'Sentence you've never seen before: what is one + one??', correct result 2, prediction 0.3799999952316284
Input sentence 'This is a sentence you've never seen before. What is negative one plus negative two?', correct result -3, prediction 6.150000095367432
```

**Observations**:
- Struggles with "one + one" despite being theoretically in the training set (predicts 3.98-5.88 instead of 2)
- Small variations in phrasing ("What is" vs "Can you answer") cause large prediction swings
- Completely fails on negative numbers (never seen during training) - predicts 6.15 instead of -3

### Two Types of Generalization

The phrase "the model memorizes the training distribution but doesn't truly understand arithmetic" conflates two distinct capabilities:

**Type (a): Distractor text generalization**
- Can the model handle *novel sentence openings* (e.g., "The capital of France is Paris, but anyway, five plus three =") when it has seen those numbers/operations before?
- This tests whether it learned to **extract mathematical operations from arbitrary natural language**
- Evidence: Model handles validation openings (different from training) reasonably well

**Type (b): Mathematical concept generalization**
- Can the model handle *novel number/operation combinations* (e.g., numbers 50-60, negative numbers, multiplication)?
- This tests whether it learned **arithmetic as a generalizable concept** vs. a lookup table for seen combinations
- Evidence: Completely fails on negative numbers; struggles even with "one + one" (possibly due to rarity in training)

### Experimental Results

Running systematic evaluation on a trained model revealed a clear distinction:

**Type (a) - Distractor text generalization: ✅ WORKS**
- Baseline error (in-distribution): 2.19 ± 2.24
- Novel similar distractors: 1.93 ± 2.03 (comparable to baseline, no significant degradation)
- Very different distractors: 4.87 ± 3.75 (only ~2x worse, despite unusual text)

**Type (b) - Mathematical concept generalization: ❌ FAILS**
- First or second number OOD (50-60): ~19-20 error (~10x worse than baseline!)
- Both numbers OOD (50-60): 25.24 ± 20.21 (complete breakdown)
- Negative numbers: 13.42 ± 11.15 (never seen in training, catastrophic failure)
- Combined OOD (distractor + numbers): 51.66 ± 14.37 (worst case)

**Conclusion**: The model successfully learned to **extract mathematical operations from variable natural language** (type a), but **failed to learn arithmetic as a generalizable concept** (type b). This supports a "memorized lookup table" hypothesis: the model memorizes addition/subtraction for number pairs in the 0-49 range, but wraps this lookup table in a surprisingly robust natural language parser that can handle novel phrasings and distractors.

**Caveat**: Very different distractors (emojis, etc.) often contained characters outside the training vocabulary, causing many test cases to fail (N=6 out of 20 succeeded).

### Systematic Generalization Evaluation

To rigorously test this, we've added `evaluate_generalization.py`, which systematically evaluates the model on 8 scenarios:

1. **Fully in-distribution**: Numbers 0-49, training openings (baseline)
2. **Novel distractor (similar)**: New openings like "Tell me: ", "Calculate " with in-dist numbers
3. **Very different distractor**: Completely unrelated text (emojis, lorem ipsum) with in-dist numbers
4. **First number OOD**: Numbers 50-60 for first operand only
5. **Second number OOD**: Numbers 50-60 for second operand only
6. **Both numbers OOD**: Both operands in 50-60 range
7. **Negative numbers**: Extreme OOD (never seen in training)
8. **Combined OOD**: Novel distractor text + OOD numbers

This evaluation generates:
- Mean absolute error for each scenario
- Predicted vs. actual scatter plots (color-coded by OOD type)
- Summary statistics comparing type (a) vs type (b) generalization
- Saved to `generalization_results.csv` and `generalization_analysis.png`

Run it via: `python fit_model.py` (automatically runs after training)

## Ideas for Future Exploration

### 1. Modern Architecture: Character-Level Transformer

**Goal**: Replace GRU with attention-based architecture while keeping it laptop-friendly.

**Approach**:
- Minimal transformer: 2-4 layers, 4-8 attention heads, ~128 hidden dims
- Keep character-level input (no tokenization)
- Positional encoding for sequence position awareness
- Still predict regression outputs initially to compare apples-to-apples

**Why**: Transformers may better capture long-range dependencies (e.g., seeing "plus" and both numbers simultaneously rather than sequentially through GRU hidden state).

**Laptop constraints**: Limit model size to ~1-5M parameters, batch size of 16-32.

### 2. GPU Acceleration

**Goal**: Get TensorFlow to use local GPU for faster training.

**Tasks**:
- Check GPU availability: `nvidia-smi` and `tf.config.list_physical_devices('GPU')`
- Install CUDA/cuDNN if needed (or use tensorflow-gpu Docker image)
- Verify GPU is being used during training
- Benchmark: CPU vs GPU training time per epoch

**Why**: Even a consumer GPU (GTX/RTX series) can speed up training 5-10x, enabling faster experimentation.

### 3. Generative Output: Character-Level Language Model

**Goal**: Convert from regression to text generation - predict answer character-by-character.

**Current limitation**: The model outputs a single float (regression). To answer "What is one plus one?", it predicts `2.0` (a number), not the text `"two"` or `"2"`.

**Approach**:
- Keep character-level input encoding (no change)
- Change output to character-level sequence generation
- Teacher forcing: given "What is one + one = ", train model to predict "t", "w", "o" sequentially
- Loss: Cross-entropy over character vocabulary at each position
- Inference: Autoregressive generation (or beam search)

**Two variants to try**:
- **Variant A**: Input includes answer, predict next char
  - Input: "What is one + one = " → Output: predict "t" → Input: "What is one + one = t" → Output: predict "w", etc.
- **Variant B**: Sequence-to-sequence
  - Encoder reads "What is one + one =", decoder generates "two"

**Why**: This is closer to how modern LLMs work, and may generalize better (e.g., can learn to output "2" or "two" or "2.0").

### 4. Training Visualization

**Goal**: Add plots to understand training dynamics.

**Currently**: Only raw text output from Keras during training - no plots or saved history.

**What to add**:
- Plot training vs validation loss over epochs (from `history` object returned by `model.fit()`)
- Separate plots for each output head (`result_loss` and `first_num_loss`)
- Save plots to file for later comparison across experiments
- Optional: Real-time plotting during training (matplotlib + IPython, or TensorBoard)
- Optional: Learning rate schedule visualization
- Optional: Prediction error distribution histogram

**Tools**: matplotlib, seaborn, or TensorBoard callback

**Why**: Essential for diagnosing overfitting, underfitting, and optimal stopping point. Currently flying blind!

### 5. Data Augmentation: More Diverse Synthetic Training Data

**Current limitations**:
- Only 35 training openings (fixed set of distractor phrases)
- Character vocabulary limited to ~50-60 chars (only those appearing in openings + number words)
- Numbers restricted to 0-49
- Only 525,000 total possible unique training sentences

**Problem with unseen characters**: When the model encounters a character not in the training vocabulary (e.g., emojis, accented letters), the LabelEncoder raises an error and that test case fails completely. This is why only 6/20 "very different distractor" test cases succeeded - the rest contained characters like 🤖, ñ, or other symbols.

**Proposed improvements**:

*For Type (a) generalization (distractor text):*
- Generate thousands of random openings programmatically (templates + variations)
- Add random punctuation (., !, ?, ..., etc.)
- Include typos and misspellings
- Add numbers-as-words in the distractor text (e.g., "Seven eight nine...")
- Expand character vocabulary to include common special characters
- Vary capitalization

*For Type (b) generalization (mathematical concepts):*
- Expand number range during training (e.g., -50 to 100 instead of 0-49)
- Include negative numbers in training distribution
- Add multiplication and division
- Include decimal results
- Vary number representations (mix "five", "5", "five point zero")

**Why this might help**:
- More diverse distractors → better type (a) generalization (learns to ignore *any* irrelevant text, not just the 35 training phrases)
- Wider number range → potentially better type (b) generalization (model might learn interpolation, though likely still won't learn true arithmetic)
- Bigger character vocabulary → fewer test case failures on unusual inputs

**Implementation**: Relatively low-effort since we already have a synthetic data generator (`simulate_sentence()`). Just need to expand the sampling distributions in `constants.py`.

### 6. Other Ideas Worth Exploring

- **Curriculum learning**: Start with single-digit addition, gradually add subtraction, larger numbers, multiplication
- **Attention visualization**: For transformers, visualize which characters the model attends to when making predictions
- **Adversarial examples**: Find minimal perturbations that break the model (e.g., "What is one + one =" vs "What is one + one?")
- **Multi-operation**: "What is (two plus three) times four ="
- **Symbolic vs numeric answers**: Train model to output either "2" or "two" and compare

### 7. Scientific Questions

- Does character-level processing help or hurt compared to word/token-level?
- Can we achieve better generalization with less data using inductive biases (e.g., attention to numbers)?
- What's the minimum model size needed to learn single-digit arithmetic reliably?

---

**Next steps**: Pick one direction and iterate. The generative output (idea #3) is particularly interesting as it bridges classical seq2seq and modern LLM approaches.