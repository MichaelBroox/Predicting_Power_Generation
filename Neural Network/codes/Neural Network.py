#!/usr/bin/env python
# coding: utf-8

# ### Loading Necessary Modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import model_selection

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler

import tensorflow as tf

import helper


# ### Styling Tables

# In[2]:


get_ipython().run_cell_magic('HTML', '', "<style type='text/css'>\ntable.dataframe th, table.dataframe td{\n    border: 3px solid purple !important;\n    color: solid black !important;\n}\n</style>")


# ### Loading Data

# In[3]:


# Loading dataset
filename = "Clean_Akosombo_data.csv"
akosombo = helper.load_csv_data(filename)


# ### Splitting Data

# In[4]:


# Splitting dataset
target_variable = "generation"
X, y, X_train, X_test, X_val, y_train, y_test, y_val = helper.split_data(akosombo, target_variable, validation_data=True)


# ### Scaling Data

# In[5]:


# Data Scaling
X_train, X_test, X_val = helper.scale(X_train, X_test, X_val, scale_validation=True)


# ### Model Creation

# In[9]:


# Creating Sequential Model
neural_network_model = Sequential()

# Input Layer 
neural_network_model.add(Dense(20, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))

# Hidden Layers
neural_network_model.add(Dense(40, kernel_initializer='normal', activation='relu'))
neural_network_model.add(Dense(40, kernel_initializer='normal', activation='relu'))
neural_network_model.add(Dropout(0.01))

# Output Layer
neural_network_model.add(Dense(1, kernel_initializer='normal', activation='linear'))

# Compiling the network
neural_network_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error', 'mean_absolute_error'])
neural_network_model.summary()


# ### Callbacks

# In[10]:


checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.h5'


checkpoint = ModelCheckpoint(checkpoint_name,
                             monitor="val_loss",
                             save_best_only=True,
                             verbose=1,
                            mode='auto')


earlystop = EarlyStopping(monitor='val_loss', 
                          min_delta=0, 
                          patience=3,
                          verbose=1,
                          mode='auto',
                          restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              mode='auto',
                              min_delta=0.00001)

# Putting call backs into a callback list
callbacks = [earlystop, checkpoint, reduce_lr]


# ### Training Model

# In[11]:


epochs = 150
batch_size = 5

history = neural_network_model.fit(
    X_train, y_train, 
    batch_size=batch_size,
    validation_data=(X_test, y_test), # Overides validation_split argument.
#     validation_split=0.25,
    epochs=epochs, 
    verbose=2,
    callbacks=callbacks,
)


# ### Visualizing Losses

# In[12]:


training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = history.epoch

fig = plt.figure(figsize=(5,5), dpi=100)

plt.plot(epochs, training_loss, label='Training loss', color='blue')
plt.plot(epochs, validation_loss, label='Validation loss', color='green')

plt.xlabel('Epochs', fontsize=15)
plt.ylabel('Loss', fontsize=15)

plt.grid(b=True, which="both", axis="both", color="black", linewidth=0.4)
plt.legend(loc='best', fontsize='medium', numpoints=1, frameon=True, shadow=True, fancybox=True)

plt.savefig('Neural_Network_Loss.png', dpi=300, transparent=True)

plt.show()


# In[13]:


training_accuracy = history.history['mean_absolute_error']
validation_loss = history.history['val_mean_absolute_error']
epochs = history.epoch


fig = plt.figure(figsize=(5,5), dpi=100)

plt.plot(epochs, training_accuracy, label='Training mae', color='blue')
plt.plot(epochs, validation_loss, label='Validation mae', color='green')

plt.xlabel('Epochs', fontsize=15)
plt.ylabel('MAE', fontsize=15)

plt.grid(b=True, which="both", axis="both", color="black", linewidth=0.4)
plt.legend(loc='best', fontsize='medium', numpoints=1, frameon=True, shadow=True, fancybox=True)

plt.savefig('Neural_Network_mean_absolute_error.png', dpi=300, transparent=True)

plt.show()


# ### Model History Data

# In[39]:


history_data = pd.DataFrame(history.history)
history_data['epochs'] = history.epoch
history_data.to_csv("neural_network_history.csv", index=True)
history_data


# In[40]:


model_parameters = pd.DataFrame(history.params)
model_parameters.to_csv("neural_network_parameters.csv", index=True)
model_parameters


# ### Model Evaluation

# In[16]:


model_name = 'Neural Network'
helper.evaluate(X_test, y_test, model_name, neural_network_model)


# ### Prediction

# In[17]:


y_pred = neural_network_model.predict(X_test)


# In[18]:


model_prediction_results = pd.DataFrame({
    'actual_generation' : list(y_test),
    'predicted_generation' : list(y_pred),
})


# In[19]:


model_prediction_results.head()


# In[20]:


model_prediction_results.dtypes


# In[21]:


# data['y_pred_generation'].str.replace('[\[\]]', '')


# In[22]:


model_prediction_results = model_prediction_results.astype({'predicted_generation':'float64'})


# In[23]:


model_prediction_results.head(10)


# In[24]:


model_prediction_results.to_csv("neural_network_predicted_values.csv", index=True)


# ### Saving the Model

# In[25]:


#Saving model
neural_network_model.save('nueral_network_model.h5')


# ### Optimizing the Hyperparameter of the Neural Network with `GridSearchCV`

# In[26]:


# Setting Random Seed
np.random.seed(82)


# In[27]:


def neural_network_regressor_model(optimizer, activation):
    model = Sequential()
    
    # Input Layer 
    model.add(Dense(20, input_dim=X_train.shape[1], kernel_initializer='normal', activation=activation)) 
    
    # Hidden Layers
    model.add(Dense(40, kernel_initializer='normal', activation=activation))
    model.add(Dense(40, kernel_initializer='normal', activation=activation))
    
    # Output Layer
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    
    # Compiling the network :
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_squared_error', 'mean_absolute_error'])
    
    print(model.summary())
    return model


# In[28]:


epochs = 150
batch_size = 5

nn_model = KerasRegressor(
    build_fn=neural_network_regressor_model, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=2
)


# In[ ]:





# In[29]:


# Kfold with with n_splits = 5 to split the Dataset into 5-folds
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=23)

# Dictionary of parameters to optimize
parameters = {
    "activation" : ['tanh', 'relu', 'elu', 'selu'], 
    "optimizer" : ['adam', 'rmsprop'],
    "batch_size" : [5, 10, 15, 20],
    "epochs" : [50, 100, 150],
}

# Scoring Metric
scorer = "r2"

# Instantiating Search object
grid = model_selection.RandomizedSearchCV(
    estimator=nn_model, 
    param_distributions=parameters, 
    scoring=scorer, 
    cv=kfold, 
    n_jobs=1, 
    verbose=2,
)

# Fit the grid object on Training Dataset
grid.fit(X_train, y_train)


# In[30]:


# Saving Hyperparameter optimization results as a DataFrame
results = pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
results.sort_values("rank_test_score", inplace=True)
results.to_csv("neural_network_hyperparameter_optimization_results.csv", index=True)
results


# ### Evaluating Best Estimator

# In[31]:


best_estimator = grid.best_estimator_
helper.evaluate(X_test, y_test, "Neural Network", best_estimator)


# ### Predicting with the Best Estimator and Saving Predicted Results as `csv`

# In[34]:


tune_y_pred = best_estimator.predict(X_test)

hyp_tune_data = pd.DataFrame(
    {"actual_generation": list(y_test), "predicted_generation": list(tune_y_pred),}
)

hyp_tune_data.to_csv("best_estimator_predicted_values.csv", index=True)

hyp_tune_data.head(10)


# In[ ]:





# ### Saving the Model

# In[35]:


import joblib

joblib.dump(best_estimator, "optimized_neural_network_model.joblib")


# In[ ]:





# In[38]:


best_estimator.get_params()


# In[ ]:




