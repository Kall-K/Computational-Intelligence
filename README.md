# Computational Intelligence (CI)

## Data Preparation

Follow the steps outlined in the [Google Machine Learning Guide](https://developers.google.com/machine-learning/guides/text-classification/step-3) to prepare your data.

### Running data_preprocessing.py

Execute the following command to run the data preprocessing script, which will create a CSV file with the input and output data needed to train the ANN:
```
python data_preprocessing.py
```
## Running ann.py

`ann.py` is the main program for training the artificial neural network (ANN). Follow these steps to run it:

1. Ensure that you have completed the data preparation step and have the CSV file ready (preprocessed_data.csv).

2. Execute the following command to run the main program:
```
python ann.py
```
3. After running `ann.py`, you will obtain four figures (into the folder named 'figures') displaying the loss at each fold and a text file containing the results (RMSE) and the parameters used to train the ANN.

