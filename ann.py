import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
import tensorflow as tf
from keras.optimizers import SGD
from keras.regularizers import l2
import matplotlib.pyplot as plt


# Create folder to save the figures about the loss convergence
directory_path = 'figures/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Create txt file to save the results of rmse
if not os.path.exists('Results.txt'):
    f = open('Results.txt', 'x')
else:
    # os.remove('Results.txt') # Use it to delete file 
    f = open('Results.txt', 'a')

# Number of nodes in hidden layer
NUM_OF_NODES = 20
# Number of hidden layers
NUM_OF_LAYERS = 1
# Dimension of input
X_DIM = 1001
# Learning rate of model
LEARNING_RATE = 0.001
# Momentum of model
MOMENTUM = 0.2
# Probability to drop nodes in the input layer
RI = 0.2
# Probability to drop nodes in the hidden layer
RH = 0.8

# Function to normalize dates to [0,1]
def normalize_date_range(date_ranges):
    #find the min date & the max date of the pairs with dates
    min_date = min(min(pair) for pair in date_ranges)
    max_date = max(max(pair) for pair in date_ranges)
    
    # normalize each pair to [-1,1]
    # return [((start - min_date) / (max_date - min_date) * 2 - 1, 
    #          (end - min_date) / (max_date - min_date) * 2 - 1)
    #         for start, end in date_ranges]
    return np.array([((start - min_date) / (max_date - min_date), 
             (end - min_date) / (max_date - min_date))
            for start, end in date_ranges])

# Read dataset 
dataset = np.loadtxt('preprocessed_data.csv', delimiter='\t', skiprows=(1))
input=StandardScaler()
dataset = input.fit_transform(X=dataset)
# Split into input and output
X = dataset[:, :-2]
Y = dataset[:, -2:]
# Normalization to [0,1]
Y = normalize_date_range(Y)
X = MinMaxScaler().fit_transform(X)
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
        model.add(Dense(NUM_OF_NODES, activation='leaky_relu', kernel_regularizer=l2(0.0001))) 
        model.add(Dropout(RH))

    # # Use the following to get different number of nodes for each hidden layer
    # model.add(Dense(NUM_OF_NODES, activation='leaky_relu', kernel_regularizer=l2(0.0001)))
    # model.add(Dense(int(NUM_OF_NODES*2), activation='leaky_relu', kernel_regularizer=l2(0.0001)))
    # model.add(Dense(int(NUM_OF_NODES*5), activation='leaky_relu', kernel_regularizer=l2(0.0001)))

    # Output layer
    model.add(Dense(1, activation='linear'))

    # Compile model
    optimizer = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=False)
    model.compile(loss=rmse, optimizer=optimizer, metrics=[rmse])

    # Fit model
    history = model.fit(X[train], Y[train], epochs=150, batch_size=100, verbose=0)
  
    plt.figure()
    plt.plot(history.history['loss'])
    plt.title(f'Nodes:{NUM_OF_NODES} ,Hidden Layers:{NUM_OF_LAYERS}, Fold:{i}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train'], loc='upper right')
    # Determine the name of the figure fold[i]->fig_i
    fig_path = os.path.join(directory_path, f'fig_{i}.png')
    plt.savefig(fig_path)
    
    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    rmseList.append(scores[0])
    print('Fold:', i, ' RMSE:', scores[0])
    f.write(f'Fold: {i} RMSE: {scores[0]} \n')



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
