import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import tensorflow as tf

# Read dataset 
# dataset = np.loadtxt("chd.csv", delimiter=",", skiprows=(1)) # didnt work for my csv
dtype = [list, int, int, int]
dataset = np.genfromtxt('new_data.csv', delimiter='\t', skip_header=1, dtype=dtype)
print(dataset[1])
# Features normalization
# norm_dataset = StandardScaler().fit_transform(X=dataset)
# print(norm_dataset)

# Split into input and output
# X = norm_dataset[:, 0]
# Y = norm_dataset[:, 1]
X = dataset[:]
Y = dataset[:, 1]
# print(X)
print(
    "\n"
)
print(Y)

# Split the data to training and testing data 5-Fold
kfold = KFold(n_splits=5, shuffle=True)
rmseList = []
rrseList = []

for i, (train, test) in enumerate(kfold.split(X)):
    # Create model
    model = Sequential()

    # model.add(Dense(10, activation="relu", input_dim=8))
    model.add(Dense(10, activation="relu", input_shape=(1000,)))
    # model.add(Dense(1, activation="linear", input_dim=10))
    model.add(Dense(1, activation="linear"))


    # Compile model
    def rmse(y_true, y_pred):
        # check here if the y_pred is into the range (date_min, date_max)
        # if this is true then the error is 0
        # else the error must be calculated by the following line of code 
        return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))

    #optimizer = keras.optimizers.SGD(lr=0.08, momentum=0.2, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer='sgd', metrics=[rmse])

    # Fit model
    model.fit(X[train], Y[train], epochs=500, batch_size=500, verbose=0)

    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    rmseList.append(scores[0])
    print("Fold :", i, " RMSE:", scores[0])

print("RMSE: ", np.mean(rmseList))

