from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import transformers
from collections import Counter

def adjust_labels_for_sep_tokens(tokenized_inputs):
    """
    Adjusts the labels for tokens following the [SEP] token in the input IDs to -100.
    Args:
        tokenized_inputs (dict): A dictionary containing the tokenized inputs, with keys including 'input_ids' and 'labels'.
    Returns:
        dict: The modified dictionary
    """
    # This function is retrieved from ChatGPT4 with prompting the function below 'tokenize_and_align_labels'
    # and asking to represent all tokens after first [SEP] token as special token with labels -100
    # Iterate over each example in the batch
    for i, input_ids in enumerate(tokenized_inputs['input_ids']):
        # Find the index of the first [SEP] token (102)
        # We start from index 1 to skip the initial [CLS] token
        first_sep_index = input_ids.index(102, 1)

        # Update labels to -100 for the first [SEP] and all following tokens
        labels = tokenized_inputs['labels'][i]
        tokenized_inputs['labels'][i] = [
            label if idx < first_sep_index else -100
            for idx, label in enumerate(labels)
        ]
    return tokenized_inputs

def tokenize_and_align_labels(examples, tokenizer, max_length, label_all_tokens=True):
    """
    Tokenizes text inputs and aligns the labels with the tokens, adjusting labels for subwords and special tokens. 

    Args:
        examples (dict): A dictionary containing the examples to be tokenized, with keys like 'token' and 'argument'.
        tokenizer (AutoTokenizer): The tokenizer used for tokenizing the text.
        max_length (int): The maximum length of the tokenized input sequences.
        label_all_tokens (bool, optional): Whether to label all subtokens with the label of the corresponding word. Defaults to True.

    Returns:
        dict: The dictionary containing the tokenized inputs with aligned labels.
    """
    # Tokenizing the input text. 'truncation=True' ensures inputs fit model max size.
    # 'is_split_into_words=True' indicates the input is already split into words.
    tokenized_inputs = tokenizer(examples["token"], truncation=True, is_split_into_words=True, padding='max_length', max_length=max_length)

    labels = []  # Initialize a list to store aligned labels for each tokenized input.
    all_word_ids = []
    # Iterate over each example. 'enumerate' provides a counter 'i' and the 'label' (negation scope).
    for i, label in enumerate(examples["argument"]):
        # Get word IDs for each token to map them back to their respective words.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None  # Initialize a variable to track the previous word index.
        label_ids = []  # List to hold the labels for each token in the current example.

        # Iterate over each word ID in 'word_ids'.
        for word_idx in word_ids:
            # Assign -100 to special tokens (word ID is None), which are ignored in loss calculation.
            if word_idx is None:
                label_ids.append(-100)
            # Assign the label to the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For subsequent tokens of the same word:
            else:
                # If 'label_all_tokens' is True, use the same label; otherwise, use -100.
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx  # Update the previous word index.

        labels.append(label_ids)  # Add the list of labels for this example to the main list.
        all_word_ids.append(word_ids)
    
    # Add the aligned labels to the tokenized inputs.
    tokenized_inputs["labels"] = labels
    tokenized_inputs['word_ids'] = all_word_ids 
    tokenized_inputs = adjust_labels_for_sep_tokens(tokenized_inputs)
    return tokenized_inputs  # Return the tokenized inputs with aligned labels.

def find_max_length_across_datasets(dataset_dict):
    """
    Finds the maximum length of tokens across all datasets in a given dictionary of datasets. This is useful for determining the padding length needed when processing the data.
    Args:
        dataset_dict (dict): A dictionary where keys are dataset names and values are the datasets themselves.
    Returns:
        int: The maximum token length found across all datasets.
    """
    # This function is added to the block in order to maintain a consistent max_length for padding.
    max_length = 0
    for dataset_name, dataset in dataset_dict.items():
        for example in dataset:
            # 'token' is assumed to be the key for tokenized text
            length = len(example["token"])
            if length > max_length:
                max_length = length
    return max_length

def set_trainer(model_path, dataset):
    """
    Sets up the trainer for fine-tuning a token classification model, including the tokenizer, model, and training arguments.
    Args:
        model_path (str): The path to the pre-trained model or its identifier from the model hub.
        dataset (DatasetDict): The dataset dictionary containing 'train', 'validation', and 'test' datasets.
    Returns:
        tuple: A tuple containing the tokenizer, trainer, and tokenized datasets.
    """
    # Setting up the task and model for natural language processing:
    # - 'model_checkpoint' is set via model_path argument.
    # 'distilbert-base-uncased' is used to fine-tune the model, which is a lightweight
    #  version of the BERT model and is not case-sensitive.
    # - 'batch_size' is set to 16, defining the number of samples to work through
    task = "srl" 
    model_checkpoint = model_path
    
    batch_size = 16

    # Initializing the tokenizer with a pre-trained model.
    # 'model_checkpoint' specifies the pre-trained model to use.
    # This tokenizer will be used to convert text into tokens that the model can understand.
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Asserting that the 'tokenizer' object is an instance of PreTrainedTokenizerFast.
    # This assertion ensures that the tokenizer has the desired properties and functionalities
    # of the PreTrainedTokenizerFast class, which is optimized for speed and efficiency.
    # If the assertion fails, it will raise an AssertionError.
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    label_all_tokens = True

    # We find the longest input throughout the dataset and keep that as the input length
    max_length = find_max_length_across_datasets(dataset)
    
    # Map the 'tokenize_and_align_labels' function to the 'dataset_dict' with batch processing (batched=True).
    # Apply the tokenize_and_align_labels function with an additional tokenizer argument
    tokenized_datasets = dataset.map(lambda examples: tokenize_and_align_labels(examples, tokenizer=tokenizer, max_length=max_length, label_all_tokens=label_all_tokens), batched=True)

    # Extracting the list of Arguments from the 'hf_training_dataset'.
    # This is done by accessing the 'argument' feature in the dataset's features,
    # and then retrieving the names associated with this feature.
    # The result is stored in 'label_list', which will contain all the unique Arguments
    # used in the training dataset.
    label_list = dataset["train"].features["argument"].feature.names

    # Create a model instance using 'AutoModelForTokenClassification' and load the pretrained model using 'from_pretrained'.
    # 'model_checkpoint' should contain the model name or path to the pretrained model.
    # 'num_labels' is the number of unique labels for your token classification task, obtained from the features.
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list))

    # Extract the model name from the 'model_checkpoint' path.
    model_name = model_checkpoint.split("/")[-1]

    # Create a TrainingArguments object to configure the fine-tuning process.
    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",  # Specify the output directory for the fine-tuned model.
        evaluation_strategy="epoch",        # Evaluation strategy, in this case, "epoch".
        learning_rate=1e-4,                # Learning rate for fine-tuning.
        per_device_train_batch_size=batch_size,  # Batch size for training on each device.
        per_device_eval_batch_size=batch_size,   # Batch size for evaluation on each device.
        num_train_epochs=1,                # Number of training epochs.
        weight_decay=0.01,                 # Weight decay for regularization.
        push_to_hub=False,                 # Whether or not to push the model to the Hugging Face Model Hub.
    )

    # After this code block, the 'args' object contains the configuration for the fine-tuning process.

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset= tokenized_datasets["train"].remove_columns(['argument']), # Remove the 'argument' column from the training dataset since we already have the labels in 'labels'.
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    return tokenizer, trainer, tokenized_datasets

def prediction_output(trainer, tokenizer, tokenized_dataset, argument_label_mapping, output_path):
    """
    Generates predictions for the test dataset, processes the predictions to skip special tokens, writes the predictions along with the gold labels and tokens to an output file.
    Args:
        trainer (Trainer): The Trainer object used for predictions.
        tokenizer (AutoTokenizer): The tokenizer used for decoding the tokens.
        tokenized_dataset (Dataset): The tokenized test dataset.
        argument_label_mapping (dict): A mapping from argument labels to their corresponding indices.
        output_path (str): The file path where the predictions and gold labels will be written.
    Returns:
        None
    """
    # Perform prediction
    predictions, labels, _ = trainer.predict(tokenized_dataset["test"])
    predictions = np.argmax(predictions, axis=2)

    label_map = {i: label for i, label in enumerate(trainer.model.config.id2label)}
    id2label = {v: k for k, v in argument_label_mapping.items()}

    actual_predictions, actual_labels = [], []
    for p, t in zip(predictions, tokenized_dataset["test"]):
        preds = []
        trues = []
        pred = []
        word_idx = 0
        for token_pred, id in zip(p, t['word_ids']):
            if id is None:
                continue            
            if id != word_idx:
                # Prediction is set to the first Argument found in the subtokens, otherwise '_'
                for p in pred:
                    ac_p = id2label[label_map[p]]
                    if ac_p != '_':
                        break

                preds.append(ac_p)
                pred = [token_pred]
                trues.append(id2label[label_map[t['argument'][word_idx]]])
                word_idx = id
            else:
                pred.append(token_pred) 

            #print(len(trues), len(preds))
        for p in pred:
            ac_p = id2label[label_map[p]]
            if ac_p != '_':
                break

        preds.append(ac_p)
        pred = [token_pred]
        trues.append(id2label[label_map[t['argument'][word_idx]]])
        
        
        actual_labels.append(trues)
        actual_predictions.append(preds)

    assert(len(actual_labels), len(actual_predictions))
    
    with open(output_path, "w") as writer:
        for sentence_id, (token_row, pred_row, label_row) in enumerate(zip(tokenized_dataset["test"], actual_predictions, actual_labels)):
            for token_id, (token, pred, label) in enumerate(zip(token_row['token'], pred_row, label_row), start=1):
                if '[PRED]' in token:
                    token = token.replace('[PRED] ', '')
                
                if '[SEP]' in token:
                    break

                writer.write(f"{sentence_id+1}\t{token_id}\t{token}\t{pred}\t{label}\n")
            writer.write("\n")  # Separate sentences by a newline