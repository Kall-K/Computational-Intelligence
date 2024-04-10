import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
# from keras import backend as K
from keras.optimizers import SGD
import matplotlib.pyplot as plt

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

# metric function rmse
def rmse(y_true, y_pred):
    min_date = y_true[:, -2]  
    max_date = y_true[:, -1]  
    avg = (min_date+max_date)/2  
 
    error = tf.where(tf.logical_and(y_pred >= min_date, y_pred <= max_date), 0.0, tf.sqrt(tf.square(y_pred - avg)))
    mean_error = tf.reduce_mean(error)
    
    return mean_error        

for i, (train, test) in enumerate(kfold.split(X)):
    # Create model
    model = Sequential()

    model.add(Dense(500, activation="leaky_relu", input_dim=X_DIM))
    model.add(Dense(500, activation="leaky_relu"))
    model.add(Dense(500, activation="leaky_relu"))
    model.add(Dense(500, activation="leaky_relu"))
    model.add(Dense(500, activation="leaky_relu"))
    model.add(Dense(500, activation="leaky_relu")) #edo auxithike to rmse hidden layers=6



    model.add(Dense(1, activation="linear"))

    # Compile model
    optimizer = SGD(learning_rate=0.001, momentum=0.2, nesterov=False)
    model.compile(loss=rmse, optimizer=optimizer, metrics=[rmse])

    # Fit model
    history = model.fit(X[train], Y[train], epochs=100, batch_size=500, verbose=0)
    
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['rmse'])
    plt.title('model accuracy')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # Evaluate model
    scores = model.evaluate(X[test], Y[test], verbose=0)
    rmseList.append(scores[0])
    print("Fold :", i, " RMSE:", scores[0])


print("RMSE: ", np.mean(rmseList))
# plt.bar([1,2,3,4,5], rmseList, color='skyblue')
plt.plot(rmseList)
plt.xlabel('Fold')
plt.ylabel('RMSE')
plt.title('RMSE for each fold')
plt.show()
