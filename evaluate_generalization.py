"""
Systematic evaluation of model generalization capabilities.

Tests two orthogonal dimensions:
(a) Distractor text generalization: unseen sentence openings
(b) Mathematical concept generalization: unseen numbers/operations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from word2number import w2n

from fit_model import (
    get_characters_one_hot_encoded,
    force_sentence_to_n_chars,
    get_output_names,
)
from constants import (
    EQUALITIES,
    NUMBERS,
    OPERATIONS_MINUS,
    OPERATIONS_PLUS,
    OUTPUT_RESULT,
)


# Define different opening categories for testing
OPENINGS_IN_DISTRIBUTION = [
    "What is ",
    "Hey ",
    "Question: ",
]

OPENINGS_NOVEL_BUT_SIMILAR = [
    "Tell me: ",
    "Please compute ",
    "Answer this: ",
    "Calculate ",
]

OPENINGS_VERY_DIFFERENT = [
    "The capital of France is Paris, but anyway, ",
    "Lorem ipsum dolor sit amet, ",
    "🤖🤖🤖 Beep boop! ",
    "In a galaxy far far away, someone asked: ",
    "This sentence contains exactly seventy-three characters including spaces. ",
]


def generate_test_sentence(number1_str, number2_str, operation, opening, equals="="):
    """Generate a test sentence with specific components."""
    return f"{opening}{number1_str} {operation} {number2_str} {equals}"


def evaluate_scenario(model, character_label_encoder, scenario_name, test_cases):
    """
    Evaluate model on a list of test cases.

    Args:
        model: Trained Keras model
        character_label_encoder: LabelEncoder for characters
        scenario_name: Description of this test scenario
        test_cases: List of (sentence, correct_result) tuples

    Returns:
        DataFrame with results for each test case
    """
    output_names = get_output_names(model)
    result_index = np.where([o.startswith(OUTPUT_RESULT) for o in output_names])[0][0]

    results = []

    for sentence, correct_result in test_cases:
        # Handle characters that might not be in the training vocabulary
        try:
            # Ensure sentence length is reasonable
            sentence = force_sentence_to_n_chars(sentence, min(len(sentence), 64))

            characters_one_hot = get_characters_one_hot_encoded(
                sentence, character_label_encoder
            )
            characters_one_hot = np.expand_dims(characters_one_hot, axis=0)

            import tensorflow as tf
            prediction = model.predict(tf.convert_to_tensor(characters_one_hot), verbose=0)[
                result_index
            ][0][0]

            error = prediction - correct_result
            abs_error = abs(error)
            squared_error = error ** 2

            results.append({
                'scenario': scenario_name,
                'sentence': sentence,
                'correct': correct_result,
                'predicted': prediction,
                'error': error,
                'abs_error': abs_error,
                'squared_error': squared_error,
            })
        except Exception as e:
            # Character not in vocabulary or other error
            results.append({
                'scenario': scenario_name,
                'sentence': sentence,
                'correct': correct_result,
                'predicted': np.nan,
                'error': np.nan,
                'abs_error': np.nan,
                'squared_error': np.nan,
            })

    return pd.DataFrame(results)


def create_test_scenarios():
    """
    Create comprehensive test scenarios to probe both types of generalization.

    Returns:
        Dictionary of {scenario_name: test_cases}
    """
    scenarios = {}

    # Scenario 1: Fully in-distribution
    # - Numbers from training range (0-49)
    # - Operations from training set
    # - Openings from training set
    test_cases = []
    for _ in range(20):
        num1 = np.random.choice(NUMBERS)
        num2 = np.random.choice(NUMBERS)
        op = np.random.choice(OPERATIONS_PLUS + OPERATIONS_MINUS)
        opening = np.random.choice(OPENINGS_IN_DISTRIBUTION)

        num1_val = w2n.word_to_num(str(num1))
        num2_val = w2n.word_to_num(str(num2))

        if op in OPERATIONS_PLUS:
            result = num1_val + num2_val
        else:
            result = num1_val - num2_val

        sentence = generate_test_sentence(num1, num2, op, opening)
        test_cases.append((sentence, result))

    scenarios['1. Fully in-distribution'] = test_cases

    # Scenario 2: Novel distractor text (type a), numbers in-distribution
    test_cases = []
    for _ in range(20):
        num1 = np.random.choice(NUMBERS)
        num2 = np.random.choice(NUMBERS)
        op = np.random.choice(OPERATIONS_PLUS + OPERATIONS_MINUS)
        opening = np.random.choice(OPENINGS_NOVEL_BUT_SIMILAR)

        num1_val = w2n.word_to_num(str(num1))
        num2_val = w2n.word_to_num(str(num2))

        if op in OPERATIONS_PLUS:
            result = num1_val + num2_val
        else:
            result = num1_val - num2_val

        sentence = generate_test_sentence(num1, num2, op, opening)
        test_cases.append((sentence, result))

    scenarios['2. Novel distractor (similar style)'] = test_cases

    # Scenario 3: Very different distractor text (type a), numbers in-distribution
    test_cases = []
    for _ in range(20):
        num1 = np.random.choice(NUMBERS)
        num2 = np.random.choice(NUMBERS)
        op = np.random.choice(OPERATIONS_PLUS + OPERATIONS_MINUS)
        opening = np.random.choice(OPENINGS_VERY_DIFFERENT)

        num1_val = w2n.word_to_num(str(num1))
        num2_val = w2n.word_to_num(str(num2))

        if op in OPERATIONS_PLUS:
            result = num1_val + num2_val
        else:
            result = num1_val - num2_val

        sentence = generate_test_sentence(num1, num2, op, opening)
        test_cases.append((sentence, result))

    scenarios['3. Very different distractor'] = test_cases

    # Scenario 4: In-distribution distractor, first number OOD (type b)
    # Use numbers 50-60 (just outside training range)
    test_cases = []
    for num1_val in range(50, 61):
        num2 = np.random.choice(NUMBERS[:20])  # Keep second number small
        num2_val = w2n.word_to_num(str(num2))
        op = np.random.choice(OPERATIONS_PLUS + OPERATIONS_MINUS)
        opening = np.random.choice(OPENINGS_IN_DISTRIBUTION)

        # Convert number to words
        import inflect
        p = inflect.engine()
        num1 = p.number_to_words(num1_val)

        if op in OPERATIONS_PLUS:
            result = num1_val + num2_val
        else:
            result = num1_val - num2_val

        sentence = generate_test_sentence(num1, num2, op, opening)
        test_cases.append((sentence, result))

    scenarios['4. First number OOD (50-60)'] = test_cases

    # Scenario 5: In-distribution distractor, second number OOD
    test_cases = []
    for num2_val in range(50, 61):
        num1 = np.random.choice(NUMBERS[:20])
        num1_val = w2n.word_to_num(str(num1))
        op = np.random.choice(OPERATIONS_PLUS + OPERATIONS_MINUS)
        opening = np.random.choice(OPENINGS_IN_DISTRIBUTION)

        import inflect
        p = inflect.engine()
        num2 = p.number_to_words(num2_val)

        if op in OPERATIONS_PLUS:
            result = num1_val + num2_val
        else:
            result = num1_val - num2_val

        sentence = generate_test_sentence(num1, num2, op, opening)
        test_cases.append((sentence, result))

    scenarios['5. Second number OOD (50-60)'] = test_cases

    # Scenario 6: Both numbers OOD
    test_cases = []
    import inflect
    p = inflect.engine()

    for _ in range(20):
        num1_val = np.random.randint(50, 61)
        num2_val = np.random.randint(50, 61)
        num1 = p.number_to_words(num1_val)
        num2 = p.number_to_words(num2_val)
        op = np.random.choice(OPERATIONS_PLUS + OPERATIONS_MINUS)
        opening = np.random.choice(OPENINGS_IN_DISTRIBUTION)

        if op in OPERATIONS_PLUS:
            result = num1_val + num2_val
        else:
            result = num1_val - num2_val

        sentence = generate_test_sentence(num1, num2, op, opening)
        test_cases.append((sentence, result))

    scenarios['6. Both numbers OOD (50-60)'] = test_cases

    # Scenario 7: Negative numbers (extreme OOD for type b)
    test_cases = []
    for _ in range(20):
        num1_val = np.random.randint(-10, 0)
        num2_val = np.random.randint(-10, 0)

        # Use numeric format for negative numbers
        num1 = f"negative {p.number_to_words(abs(num1_val))}"
        num2 = f"negative {p.number_to_words(abs(num2_val))}"

        op = np.random.choice(OPERATIONS_PLUS + OPERATIONS_MINUS)
        opening = np.random.choice(OPENINGS_IN_DISTRIBUTION)

        if op in OPERATIONS_PLUS:
            result = num1_val + num2_val
        else:
            result = num1_val - num2_val

        sentence = generate_test_sentence(num1, num2, op, opening)
        test_cases.append((sentence, result))

    scenarios['7. Negative numbers (extreme OOD)'] = test_cases

    # Scenario 8: Combined OOD (both types a and b)
    test_cases = []
    for _ in range(20):
        num1_val = np.random.randint(50, 70)
        num2_val = np.random.randint(50, 70)
        num1 = p.number_to_words(num1_val)
        num2 = p.number_to_words(num2_val)
        op = np.random.choice(OPERATIONS_PLUS + OPERATIONS_MINUS)
        opening = np.random.choice(OPENINGS_VERY_DIFFERENT)

        if op in OPERATIONS_PLUS:
            result = num1_val + num2_val
        else:
            result = num1_val - num2_val

        sentence = generate_test_sentence(num1, num2, op, opening)
        test_cases.append((sentence, result))

    scenarios['8. Combined OOD (distractor + numbers)'] = test_cases

    return scenarios


def plot_generalization_results(df, output_file='generalization_analysis.png'):
    """Create visualization of generalization performance."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Mean Absolute Error by scenario
    ax = axes[0, 0]
    scenario_stats = df.groupby('scenario')['abs_error'].agg(['mean', 'std']).reset_index()
    scenario_stats = scenario_stats.sort_values('mean')

    ax.barh(scenario_stats['scenario'], scenario_stats['mean'], xerr=scenario_stats['std'])
    ax.set_xlabel('Mean Absolute Error')
    ax.set_title('Generalization Performance by Scenario')
    ax.axvline(x=5, color='red', linestyle='--', alpha=0.3, label='Error = 5')
    ax.legend()

    # 2. Box plot of errors
    ax = axes[0, 1]
    scenario_order = scenario_stats['scenario'].tolist()
    sns.boxplot(data=df, y='scenario', x='abs_error', order=scenario_order, ax=ax)
    ax.set_xlabel('Absolute Error')
    ax.set_title('Error Distribution by Scenario')
    ax.set_xlim(0, min(df['abs_error'].quantile(0.95), 50))

    # 3. Predicted vs Actual (color by scenario type)
    ax = axes[1, 0]

    # Color code: in-dist vs distractor-OOD vs number-OOD
    def categorize_scenario(scenario_name):
        if 'in-distribution' in scenario_name:
            return 'In-distribution'
        elif 'distractor' in scenario_name or 'Very different' in scenario_name:
            return 'Type (a): Distractor OOD'
        else:
            return 'Type (b): Number OOD'

    df['ood_type'] = df['scenario'].apply(categorize_scenario)

    for ood_type, color in zip(
        ['In-distribution', 'Type (a): Distractor OOD', 'Type (b): Number OOD'],
        ['green', 'blue', 'red']
    ):
        mask = df['ood_type'] == ood_type
        ax.scatter(
            df[mask]['correct'],
            df[mask]['predicted'],
            alpha=0.5,
            s=20,
            label=ood_type,
            color=color
        )

    # Perfect prediction line
    lims = [
        min(df['correct'].min(), df['predicted'].min()),
        max(df['correct'].max(), df['predicted'].max())
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('Correct Result')
    ax.set_ylabel('Predicted Result')
    ax.set_title('Predicted vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')

    summary_stats = df.groupby('ood_type').agg({
        'abs_error': ['mean', 'std', 'median'],
        'scenario': 'count'
    }).round(2)

    summary_text = "Summary by OOD Type:\n\n"
    for ood_type in ['In-distribution', 'Type (a): Distractor OOD', 'Type (b): Number OOD']:
        if ood_type in summary_stats.index:
            row = summary_stats.loc[ood_type]
            summary_text += f"{ood_type}:\n"
            summary_text += f"  Mean Abs Error: {row[('abs_error', 'mean')]:.2f} ± {row[('abs_error', 'std')]:.2f}\n"
            summary_text += f"  Median Abs Error: {row[('abs_error', 'median')]:.2f}\n"
            summary_text += f"  N = {int(row[('scenario', 'count')])}\n\n"

    ax.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top', family='monospace')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_file}")

    return fig


def run_evaluation(model, character_label_encoder):
    """Run full evaluation suite and generate visualizations."""

    print("Creating test scenarios...")
    scenarios = create_test_scenarios()

    print(f"Evaluating {len(scenarios)} scenarios...")
    all_results = []

    for scenario_name, test_cases in scenarios.items():
        print(f"  {scenario_name}: {len(test_cases)} test cases")
        df = evaluate_scenario(model, character_label_encoder, scenario_name, test_cases)
        all_results.append(df)

    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)

    # Remove any NaN rows (from vocabulary errors)
    df_all = df_all.dropna()

    # Print summary
    print("\n" + "="*80)
    print("GENERALIZATION EVALUATION RESULTS")
    print("="*80)

    for scenario_name in df_all['scenario'].unique():
        scenario_df = df_all[df_all['scenario'] == scenario_name]
        mean_error = scenario_df['abs_error'].mean()
        std_error = scenario_df['abs_error'].std()
        median_error = scenario_df['abs_error'].median()

        print(f"\n{scenario_name}")
        print(f"  Mean absolute error: {mean_error:.2f} ± {std_error:.2f}")
        print(f"  Median absolute error: {median_error:.2f}")
        print(f"  N = {len(scenario_df)}")

    # Create visualization
    print("\nGenerating plots...")
    plot_generalization_results(df_all)

    # Save detailed results
    csv_file = 'generalization_results.csv'
    df_all.to_csv(csv_file, index=False)
    print(f"Detailed results saved to {csv_file}")

    return df_all


if __name__ == "__main__":
    print("This script requires a trained model.")
    print("Run fit_model.py first, or import this module and call run_evaluation(model, encoder)")
