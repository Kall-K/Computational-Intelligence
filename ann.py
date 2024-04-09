import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
# from keras import backend as K
from keras.optimizers import SGD

# dimension of input
X_DIM = 1000

# Read dataset 
dataset = np.loadtxt("new_data.csv", delimiter="\t", skiprows=(1)) 

# Split into input and output
X = dataset[:, :-3]
Y = dataset[:, -2:]

print(dataset.shape, X.shape, Y.shape)


# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)
rmseList = []
rrseList = []

# @tf.function
def rmse(y_true, y_pred):
    min_date = y_true[:, -2]  
    max_date = y_true[:, -1]  
    avg = (min_date+max_date)/2  
 
    # within = tf.reduce_all(tf.logical_and(tf.greater_equal(y_pred, min_date),
    #                                         tf.less_equal(y_pred, max_date)))

    error = tf.where(tf.logical_and(y_pred >= min_date, y_pred <= max_date), 0.0, tf.sqrt(tf.square(y_pred - avg)))
    mean_error = tf.reduce_mean(error)
    
    return mean_error
    # if within:
    #     return 0.0
    # else:
    #     return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))          
    # return tf.sqrt(tf.reduce_mean(tf.square(y_pred - avg)))          


for i, (train, test) in enumerate(kfold.split(X)):
    # Create model
    model = Sequential()

    model.add(Dense(10, activation="relu", input_dim=X_DIM))
    model.add(Dense(1, activation="linear"))

    # Compile model
    optimizer = SGD(learning_rate=0.001, momentum=0.2, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[rmse])

    # Fit model
    model.fit(X[train], Y[train], epochs=500, batch_size=500, verbose=0)

    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    rmseList.append(scores[0])
    print("Fold :", i, " RMSE:", scores[0])

print("RMSE: ", np.mean(rmseList))

