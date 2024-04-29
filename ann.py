import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
import tensorflow as tf
from keras.optimizers import SGD
from keras.regularizers import l2
# from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Create folder to save the figures about the loss convergence
directory_path = 'figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Create txt file to save the results of rmse
if not os.path.exists('Results.txt'):
    f = open('Results.txt', 'x')
else:
    f = open('Results.txt', 'a')

# Number of nodes in hidden layer
NUM_OF_NODES = 30
# Number of hidden layers
NUM_OF_LAYERS = 1
# Dimension of input
X_DIM = 1001
# Learning rate of model
LEARNING_RATE = 0.001
# Momentum of model
MOMENTUM = 0.2
# Probability to drop out nodes of the input layer
RI = 0.2
# Probability to drop out nodes of the hidden layer
RH = 0.8
 
# Custom Function to normalize dates to [0,1]
def normalize_date_range(date_ranges):
    #find the min date & the max date of the pairs with dates
    min_date = min(min(pair) for pair in date_ranges)
    max_date = max(max(pair) for pair in date_ranges)
    
    return np.array([((start - min_date) / (max_date - min_date), 
             (end - min_date) / (max_date - min_date))
            for start, end in date_ranges])

# Custom Function to standarize dates
def standarize_date_range(date_ranges):
    dates = np.hstack(date_ranges)
    mean_value = np.mean(dates)
    deviation_value = np.std(dates)
    return (date_ranges - mean_value) / deviation_value
    
# Read dataset 
dataset = np.loadtxt('preprocessed_data.csv', delimiter='\t', skiprows=(1))
# Split into input and output
X = dataset[:, :-2]
Y = dataset[:, -2:]
# Standarization μ=0,σ=1
X = StandardScaler().fit_transform(X=X)
Y = standarize_date_range(Y)
# Normalization to [0,1]
X = MinMaxScaler().fit_transform(X=X)
Y = normalize_date_range(Y)

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)
rmseList = []
rrseList = []

# Custom RMSE function 
def rmse(y_true, y_pred):
    min_date = y_true[:, 0]  
    max_date = y_true[:, 1]  
    avg = (min_date+max_date)/2  

    error = tf.where(tf.logical_and(y_pred >= min_date, y_pred <= max_date),
                      0.0, tf.sqrt(tf.reduce_mean(tf.square(y_pred - avg))))
    return error

for i, (train, test) in enumerate(kfold.split(X)):
    # Create model
    model = Sequential()

    # Input layer
    model.add(Input(shape=(X_DIM,)))
    model.add(Dropout(RI))

    # Add hidden layers
    for _ in range(NUM_OF_LAYERS):
        model.add(Dense(NUM_OF_NODES, activation='relu', kernel_regularizer=l2(0.001))) 
        model.add(Dropout(RH))

    # # Use the following to get different number of nodes for each hidden layer # # # # 
    # model.add(Dense(NUM_OF_NODES, activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(Dense(int(NUM_OF_NODES/2), activation='relu', kernel_regularizer=l2(0.001)))
    # model.add(Dense(int(NUM_OF_NODES/5), activation='relu', kernel_regularizer=l2(0.001)))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
 
    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    optimizer = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    model.compile(loss=rmse, optimizer=optimizer, metrics=[rmse])

    # Fit model
    history = model.fit(X[train], Y[train], epochs=80, batch_size=100, verbose=0)
   
    plt.figure()
    plt.plot(history.history['loss'], label=f'train')
    plt.title(f'Nodes:{NUM_OF_NODES},Hidden Layers:{NUM_OF_LAYERS}, Fold:{i}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    # Determine the name of the figure fold[i]->fig_i
    fig_path = os.path.join(directory_path, f'fig_{i}.png')
    plt.savefig(fig_path)
    
    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    rmseList.append(scores[0])
    print('Fold:', i, ' RMSE:', scores[0])
    f.write(f'Fold: {i} RMSE: {scores[0]} \n')

    # # SAVE POINTS OF LOSS # # # # # # # # # # # # # # # # # # # # 
    # file_path= f'convergence/points{NUM_OF_LAYERS}{NUM_OF_NODES}{i}.json'
    # with open(file_path, "w") as json_file:
    #     json.dump(history.history['loss'], json_file)
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


print('RMSE: ', np.mean(rmseList))
f.write(f'RMSE: {np.mean(rmseList)} \n')

print(f'Num of nodes in hidden layer:{NUM_OF_NODES}')
print(f'Num of hidden layers:{NUM_OF_LAYERS}')
f.write(f'Num of nodes in hidden layer:{NUM_OF_NODES} \n')
f.write(f'Num of hidden layers:{NUM_OF_LAYERS} \n')
f.write(f'Learning Rate:{LEARNING_RATE}\n'
        f'Momentum:{MOMENTUM}\n'
        f'Input Layer Dropout:{RI}\n'
        f'Hidden Layer Dropout:{RH}\n')
f.write('---------------------------------\n')
f.close()
