import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

antenna = pd.read_csv("/content/antenna.csv") #Kaggle link to the dataset: https://www.kaggle.com/datasets/renanmav/metamaterial-antennas
columns_with_blank = antenna.columns[antenna.isnull().any()]

#Drops rows having blank values
antenna_cleaned = antenna.dropna(subset=columns_with_blank)

X = antenna_cleaned.drop("s", axis=1)
y = antenna_cleaned["s"]

#Train-Test split
X_train = X[:500]
y_train = y[:500]
X_test = X[500:]
y_test = y[500:]

#Set random seed
tf.random.set_seed(42)

#Sequential model
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation='linear'),
  tf.keras.layers.Dense(1)
])

#Compile model
model.compile(loss=tf.keras.losses.mae, #mae is mean absolute error
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["mae"])

#Fit model
model1=model.fit(X, y, epochs=500)

y_pred = model.predict(X_test)

#Loss/MAE vs. Epochs plot
pd.DataFrame(model1.history).plot()
plt.ylabel("loss") #Curve for loss and MAE will overlap because loss function and Metric is same-'MAE'
plt.xlabel("epochs")

X_test_tensor = tf.constant(X_test.values, dtype=tf.float32) #Creates a constant tensor containing the values from your X_test with the data type set to 32-bit floating-point numbers
row_2 = X_test_tensor[0:9, :] 
y_pred1 = model.predict(row_2)

y_pred1_tensor = tf.convert_to_tensor(y_pred1) #Converts array to tensor
y_pred1_tensor_squeeze=tf.squeeze(y_pred1_tensor) #For the sake to shape similarity

X_list_test=tf.range(0,9,1)

#To compare the model predicted output with the test output
plt.plot(X_list_test,y_test)
plt.plot(X_list_test,y_pred1_tensor_squeeze,'r')
