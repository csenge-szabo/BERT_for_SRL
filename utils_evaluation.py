from collections import defaultdict
from sklearn.metrics import classification_report

def evaluate_model_from_file(file_path):
    """
    Reads data from a TSV file, evaluates model predictions, and prints a classification report.
    Args:
        file_path (str): Path to the TSV file containing model predictions.
    Returns: None
    """
    # Initialize a defaultdict to store sentences.
    sentences = defaultdict(list)

    # Open the file in read mode with utf-8 encoding.
    with open(file_path, 'r', encoding='utf-8') as file:
        # Iterate over each line in the file.
        for line in file:
            # Skip empty lines
            if not line.strip():
                continue
            # Split the line into a list of strings using tab as the delimiter.
            row = line.strip().split('\t')
            # If the row has at least 4 columns, process it.
            if len(row) >= 4:
                # The first column is the sentence id.
                sentence_id = row[0]
                # Create a dictionary to store token data.
                token_data = {
                    'token_id': row[1],  # The second column is the token id.
                    'token': row[2],  # The third column is the token.
                    'gold_label': row[4],  # The fifth column is the gold label.
                    'system_label': row[3]  # The fourth column is the system label.
                }
                # Append the token data to the list of tokens for the current sentence id.
                sentences[sentence_id].append(token_data)
    # Initialize two lists to store the true labels and the predicted labels.
    y_true = []
    y_pred = []

    # Iterate over each sentence in the sentences dictionary.
    for sentence_id, tokens in sentences.items():
        # Iterate over each token in the sentence.
        for token in tokens:
            # Append the gold label to the list of true labels.
            y_true.append(token['gold_label'])
            # Append the system label to the list of predicted labels.
            y_pred.append(token['system_label'])
    # Generate a classification report using sklearn's classification_report function.
    report = classification_report(y_true, y_pred, digits=3, zero_division=0)

    print("Classification Report:")
    print(report)
    
if __name__ == "__main__":
    file_path = 'prediction_output/BERT3_predictions.tsv'
    evaluate_model_from_file(file_path)

