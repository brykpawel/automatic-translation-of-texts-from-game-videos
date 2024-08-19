import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Define a function to read the CSV file with different encodings and handle errors
def read_csv_with_encodings(file_path, encodings):
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, header=None, encoding=encoding, on_bad_lines='skip')
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file with the provided encodings.")

# List of encodings to try
encodings_to_try = ['utf-8-sig', 'latin1', 'cp1252']

# Read CSV files
candidate_df = read_csv_with_encodings('tlumaczenie_z_netfliksa.csv', encodings_to_try)
reference_df = read_csv_with_encodings('tlumaczenie_aplikacja.csv', encodings_to_try)

bleu_scores = 0
counter = 0

# Calculate BLEU scores for each sentence pair
for i in range(min(len(candidate_df), len(reference_df))):
    candidate_sentence = candidate_df.iloc[i, 0]
    reference_sentence = reference_df.iloc[i, 0]
    print(candidate_sentence)
    print(reference_sentence)

    # Split sentences into words
    candidate = candidate_sentence.split()
    reference = [reference_sentence.split()]

    # Calculate BLEU score
    score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
    print(f'BLEU score for sentence nr {i}: {score:.6f}')
    bleu_scores += score
    counter += 1

# Calculate and print the average BLEU score
average_bleu_score = bleu_scores / counter
print(f'Average BLEU score: {average_bleu_score:.6f}')