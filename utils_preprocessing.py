def transform_raw_data_v3(input_file, output_file=None):
    """
    Transform the input data to:
    - Ensure sentences are separated by a single newline, with tokens directly following each other without extra newlines.
    - Keep only columns token_id, token and argument
    - Add a unique sentence ID number as a new column at the most left.
    - Add the [PRED] to the predicate token.

    Args:
    - input_file (str): A string containing the file path to the input data.
    - output_file (str): A string containing the file path to store the output data. If none, nothing is saved.

    Returns:
    - List of sentences with a list of tokens, with a list in order: sentence_id, token_id, token, argument class

    """
    with open(input_file, mode='r', encoding='utf-8') as f:
            sentences = []  # List to store sentences
            pred_sentences = []
            pred = 0
            s_id = 1
            for line in f.readlines():
                line = line.strip('\n')

                # Start new sentence
                if line.startswith('# sent_id'):
                    s_id += pred
                    sentences.extend(pred_sentences)
                    pred_sentences = []
                    pred = 0

                # Discard non token 
                elif line.startswith('#'):
                    continue
                
                # Extract tokens
                else:
                    # Parse token information
                    row = line.strip().split('\t')
                    if len(row) >= 11:  # Ensure all needed columns are present 
                        if row[10] != '_':
                            pred += 1    
                        
                        for i in range(11,len(row)):
                            if row[0] == '1':
                                pred_sentences.append([])
                            #if s_id == 1 and pred == 1:
                            #    print('blub', pred_sentences)
                            # Extract token_id, token, and args
                            pred_sentences[i-11].append([
                                str(s_id + i-11),
                                row[0], 
                                '[PRED] ' + row[1] if row[10] != '_' and pred == i-10 else row[1], 
                                row[i] if row[i] != 'V' and row[i] != 'C-V' else '_'
                                ])
                       
            sentences.extend(pred_sentences)              
    
    # Write to a new file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                for token in sentence:
                    f.write('\t'.join(token) + '\n')
                f.write('\n')
    
    return sentences

def transform_raw_data_v1(input_file, output_file=None):
    """
    Transform the input data to:
    - Ensure sentences are separated by a single newline, with tokens directly following each other without extra newlines.
    - Keep only columns token_id, token and argument
    - Add a unique sentence ID number as a new column at the most left.
    - Add a [SEP] token and the predicate token afterwards at the end of the sentence.

    Args:
    - input_file (str): A string containing the file path to the input data.
    - output_file (str): A string containing the file path to store the output data. If none, nothing is saved.

    Returns:
    - List of sentences with a list of tokens, with a list in order: sentence_id, token_id, token, argument class

    """
    with open(input_file, mode='r', encoding='utf-8') as f:
            sentences = []  # List to store sentences
            pred_token = []
            pred_sentences = []
            pred = 0
            s_id = 1
            for line in f.readlines():
                line = line.strip('\n')

                # Start new sentence
                if line.startswith('# sent_id'):
                    # print(pred_token)
                    # Add [SEP] token and predicate token at the end of each sentence
                    for i, pred_sent in enumerate(pred_sentences):
                        ps_id, last_token_id, _, _ = pred_sent[-1]
                        pred_sentences[i].append([ps_id, str(int(last_token_id)+1), '[SEP]', '_'])
                        pred_sentences[i].append([ps_id, str(int(last_token_id)+2), pred_token[i], '_'])
                    
                    # Update pred sentences
                    sentences.extend(pred_sentences)
                    pred_sentences = []
                    pred_token = []

                    # Update counters
                    s_id += pred
                    pred = 0

                # Discard non tokens
                elif line.startswith('#'):
                    continue
                
                # Extract tokens
                else:
                    # Parse token information
                    row = line.strip().split('\t')
                    if len(row) >= 11:  # Ensure all needed columns are present 
                        if row[10] != '_':
                            pred += 1    
                        
                        for i in range(11,len(row)):
                            if row[0] == '1':
                                pred_sentences.append([])
                            
                            # Extract token_id, token, and args
                            pred_sentences[i-11].append([
                                str(s_id + i-11),
                                row[0], 
                                row[1], 
                                row[i] if row[i] != 'V' and row[i] != 'C-V' else '_'
                                ])
                            # Extract predicate token
                            if row[10] != '_' and pred == i-10:
                                pred_token.append(row[1])

            # Add [SEP] token and predicate token at the end of each sentence
            for i, pred_sent in enumerate(pred_sentences):
                ps_id, last_token_id, _, _ = pred_sent[-1]
                pred_sentences[i].append([ps_id, str(int(last_token_id)+1), '[SEP]', '_'])
                pred_sentences[i].append([ps_id, str(int(last_token_id)+2), pred_token[i], '_'])
                    
            sentences.extend(pred_sentences)              
    
    # Write to a new file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                for token in sentence:
                    f.write('\t'.join(token) + '\n')
                f.write('\n')
    
    return sentences


def transform_raw_data_v2(input_file, output_file=None):
    """
    Transform the input data to:
    - Ensure sentences are separated by a single newline, with tokens directly following each other without extra newlines.
    - Keep only columns token_id, token and argument
    - Add a unique sentence ID number as a new column at the most left.
    - Add a [SEP] token and the predicate window tokens afterwards at the end of the sentence.

    Args:
    - input_file (str): A string containing the file path to the input data.
    - output_file (str): A string containing the file path to store the output data. If none, nothing is saved.

    Returns:
    - List of sentences with a list of tokens, with a list in order: sentence_id, token_id, token, argument class

    """
    with open(input_file, mode='r', encoding='utf-8') as f:
            sentences = []  # List to store sentences
            pred_token = []
            pred_sentences = []
            pred = 0
            s_id = 1
            for line in f.readlines():
                line = line.strip('\n')

                # Start new sentence
                if line.startswith('# sent_id'):

                    # Add [SEP] token and predicate token at the end of each sentence
                    for i, pred_sent in enumerate(pred_sentences):
                        ps_id, last_token_id, _, _ = pred_sent[-1]
                        pred_sentences[i].append([ps_id, str(int(last_token_id)+1), '[SEP]', '_'])
                        pred_sentences[i].append([ps_id, str(int(last_token_id)+2), pred_sent[pred_token[i]-1][2], '_'])
                        pred_sentences[i].append([ps_id, str(int(last_token_id)+3), pred_sent[pred_token[i]][2], '_'])                        
                        pred_sentences[i].append([ps_id, str(int(last_token_id)+4), pred_sent[pred_token[i]+1][2], '_'])
                    
                    # Update pred sentences
                    sentences.extend(pred_sentences)
                    pred_sentences = []
                    pred_token = []

                    # Update counters
                    s_id += pred
                    pred = 0

                # Discard non tokens
                elif line.startswith('#'):
                    continue
                
                # Extract tokens
                else:
                    # Parse token information
                    row = line.strip().split('\t')
                    if len(row) >= 11:  # Ensure all needed columns are present 
                        if row[10] != '_':
                            pred += 1    
                        
                        for i in range(11,len(row)):
                            if row[0] == '1':
                                pred_sentences.append([])
                            
                            # Extract token_id, token, and args
                            pred_sentences[i-11].append([
                                str(s_id + i-11),
                                row[0], 
                                row[1], 
                                row[i] if row[i] != 'V' and row[i] != 'C-V' else '_'
                                ])
                            # Extract predicate token
                            if row[10] != '_' and pred == i-10:
                                pred_token.append(int(row[0])-1)

            # Add [SEP] token and predicate token at the end of each sentence
            for i, pred_sent in enumerate(pred_sentences):
                ps_id, last_token_id, _, _ = pred_sent[-1]
                pred_sentences[i].append([ps_id, str(int(last_token_id)+1), '[SEP]', '_'])
                pred_sentences[i].append([ps_id, str(int(last_token_id)+2), pred_sent[pred_token[i]-1][2], '_'])
                pred_sentences[i].append([ps_id, str(int(last_token_id)+3), pred_sent[pred_token[i]][2], '_'])                        
                pred_sentences[i].append([ps_id, str(int(last_token_id)+4), pred_sent[pred_token[i]+1][2], '_'])
                           
            sentences.extend(pred_sentences)              
    
    # Write to a new file if output_file is provided
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                for token in sentence:
                    f.write('\t'.join(token) + '\n')
                f.write('\n')
    
    return sentences

if __name__ == "__main__":
    # Specify the input and output file paths
    input_file_path = 'data/en_ewt-up-dev.conllu'  # Adjusted to the actual uploaded file path
    
    # Call the function to perform the transformation
    transform_raw_data_v1(input_file_path, 'data/bert-dev1-new.conllu')
    transform_raw_data_v2(input_file_path, 'data/bert-dev2-new.conllu')
    transform_raw_data_v3(input_file_path, 'data/bert-dev3-new.conllu')
