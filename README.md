# Fine-tuning a LLM for Semantic Role Labeling (SRL)

### Introduction
This project was carried out as part of the Advanced NLP Master's course at VU Amsterdam. The aim of this project is to fine-tune a transformer-based model, DistilBERT, for SRL using different methods of predicate indication.

### Authors: 
- Selin Acikel
- Murat Ertas
- Tessel Haagen
- Csenge Szabó

### Data: 
For this project we are utilizing the [Universal Proposition Banks version 1.0](https://universalpropositions.github.io) for English language, which was created with the aim to study Semantic Role Labelling.

### Steps before experiments: 
- Ensure that the requirements.txt listed are available. If you are using this notebook on Google Colab, add a code block for installing the additional requirements at the top of the notebook and restart the kernel after installation.
- Place the source data (training, development, test) into the 'data' subdirectory.
- Place the trained models (Model 1, Model 2, Model 3) into the 'models' subdirectory and unzip them in order to use them for predictions and evaluation. 
Link to trained models: [DropBox](https://www.dropbox.com/scl/fo/xv6pkmvqfs4eaptr0aw9i/h?rlkey=jk8ggqbkrngjclxduq2fod3dy&dl=0)

### Files: 
Please use `the_notebook.ipynb` for running the experiments. We utilized Visual Studio Code for running our experiments.

Auxiliary scripts:
- `utils_preprocessing.py`
- `utils_data_processing.py`
- `utils_model_training.py`
- `utils_evaluation.py`

Model predictions in 'prediction_output' subdirectory:
- BERT1_predictions.tsv (model 1)
- BERT2_predictions.tsv (model 2)
- BERT3_predictions.tsv (model 3)








