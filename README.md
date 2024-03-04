# NLP Coursework
NLP 2024 Coursework
This repo is for the code used in the NLP Coursework in 2024 at Imperial College London. 

Navigating the codebase:

1. Data directory:
    - Training and testing data from dontpatronizeme.
    - Notebook containing exploratory data analysis.
2. Data Preperation directory:
    - **preproc.py**: PreProcessor class which is used to apply pre processing and data augmentations to the training data.
    - **preproc_tests.ipynb**: Eperiments on different pre processing techniques.
    - **aug_tests.ipynb**: Experiments on different augmentation techniques.
    - **dont_patronize_me.py**: Data Loader. 
3. Model Directory:
    - **bert_model.py**: Model class implementing forward pass, layer freeze, hyperparameters and utility model functions.
    - **downsample_base_bert.ipynb**: Experiment with base model.
    - **downsample_base_bow.ipynb**: Experiment with a Bag of Words model.
    - **downsample_base_ngram.ipynb**: Experiment with an N-Gram model.
    - **freeze_weights_and_epoch_tests.ipynb**: Experiment with RoBERTa model with different hyperparameters and augmentations.
4. Ensemble Model Directory:
    - **ensemble_model.py**: EnsembleModel class to create an ensemble of models and calculate final predictions based on a list of models' predictions.
    - **ensemble_baseline_test.py**: Experiments with ensemble of models.

Our best chosen model is defined within **best_model_test.ipynb** which runs the model and outputs a final prediction label for the dev and test sets.

Thank you for reading!