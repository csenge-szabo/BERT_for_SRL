import csv
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Sequence, Value, Features


def load_data(file_path):
    """
    Load the TSV file into a Pandas DataFrame.

    Args:
    file_path (str): Path to the .tsv file designed for BERT input.

    Returns:
    DataFrame: A DataFrame containing the loaded data.
    # We implemented an earlier version of this function which was used for Negation Scope Detection task in Applied Text Mining Methods course
    """
    col_names=[
        'sentence_id', 'token_id', 'token', 'argument'
    ]
    # Reading data from a TSV file into a DataFrame with specified column names.
    # , lineterminator='\r'
    df = pd.read_csv(file_path, sep='\t', header=None, names=col_names, quoting=csv.QUOTE_NONE)

    # Dropping rows where any column has missing values (NA).
    df = df.dropna(how='any')
    df['token_id'] = df['token_id'].astype(str)

    return df

def transform_data(df, target_label_mapping):
    """
    Transform the DataFrame into a format suitable for machine learning models.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target_label_mapping (dict): Mapping for 'argument' labels to indices.

    Returns:
        list: A list of dictionaries, each representing a transformed data record.
    """
    # Grouping the data by 'chapter' and 'sentence_num' for processing.
    grouped = df.groupby(['sentence_id'])
    transformed_data = []


    # Transforming the grouped data into a format suitable for training.
    for sentence_num, group in grouped:
        record = {
            'sentence_id' : group['sentence_id'].tolist(),
            'token_id' : group['token_id'].tolist(),
            'token': group['token'].tolist(),
            'argument': [target_label_mapping[label] for label in group['argument'].tolist()]
        }
        transformed_data.append(record)
    return transformed_data

def create_datasets(training_file_path, test_file_path, dev_file_path):
    """
    Load data from specified file paths, transform it, and create datasets for training, validation, and testing.

    Args:
        training_file_path (str): Path to the training data file.
        test_file_path (str): Path to the test data file.
        dev_file_path (str): Path to the development (validation) data file.

    Returns:
        tuple: 
            - datasets.DatasetDict containing 'train', 'validation', and 'test' datasets.
            - dict mapping argument labels to indices, used for transforming 'argument' column in the data.
    """
    # Loading the data from the specified file paths
    training_data = load_data(training_file_path)
    test_data = load_data(test_file_path)
    dev_data = load_data(dev_file_path)

    # Extracting arguments and creating a unique argument label mapping
    training_args = training_data['argument']
    dev_args = dev_data['argument']
    test_args = test_data['argument']
    combined_arg_labels = pd.concat([training_args, dev_args, test_args])
    unique_arguments = sorted(combined_arg_labels.unique().tolist())
    argument_label_mapping = {label: idx for idx, label in enumerate(unique_arguments)}

    # Transforming the data to the required format
    transformed_training_data = transform_data(training_data, argument_label_mapping)
    transformed_test_data = transform_data(test_data, argument_label_mapping)
    transformed_dev_data = transform_data(dev_data, argument_label_mapping)

    # Defining the features for the Hugging Face dataset
    features = Features({
        'sentence_id': Sequence(feature=Value(dtype='string')),
        'token_id': Sequence(feature=Value(dtype='string')),
        'token': Sequence(feature=Value(dtype='string')),
        'argument': Sequence(feature=ClassLabel(names=unique_arguments))
    })

    # Converting the transformed data to Hugging Face datasets
    hf_training_dataset = Dataset.from_pandas(pd.DataFrame(transformed_training_data), features=features)
    hf_test_dataset = Dataset.from_pandas(pd.DataFrame(transformed_test_data), features=features)
    hf_dev_dataset = Dataset.from_pandas(pd.DataFrame(transformed_dev_data), features=features)

    # Creating a DatasetDict to hold the training, validation, and test datasets
    dataset_dict = DatasetDict({
        'train': hf_training_dataset,
        'validation': hf_dev_dataset,
        'test': hf_test_dataset
    })

    return dataset_dict, argument_label_mapping
