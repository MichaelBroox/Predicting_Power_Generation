#!/usr/bin/env python
# coding: utf-8

# ### Loading all necessary Packages/Libraries

# In[1]:


# Loading the iconic trio 🔥
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing model_selection to get access to some dope functions like GridSearchCV()
from sklearn import model_selection

# from sklearn.externals import joblib

# Loading models
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
from sklearn import ensemble
from sklearn import neighbors

# custom
import helper

# Loading black for formatting codes
get_ipython().run_line_magic('load_ext', 'blackcellmagic')


# ### Styling Tables

# In[2]:


get_ipython().run_cell_magic('HTML', '', "<style type='text/css'>\ntable.dataframe th, table.dataframe td{\n    border: 3px solid purple !important;\n    color: solid black !important;\n}\n</style>")


# ### Loading the Dataset

# In[3]:


# Loading dataset
filename = "Clean_Akosombo_data.csv"
akosombo = helper.load_csv_data(filename)


# ### Splitting the Dataset

# In[4]:


# Splitting dataset
target_variable = "generation"
X, y, X_train, X_test, y_train, y_test = helper.split_data(akosombo, target_variable)


# ### Scaling the Dataset

# In[5]:


# Data Scaling
X_train, X_test = helper.scale(X_train, X_test)


# ### Chosing Baseline Models and Training Models

# In[6]:


# Instantiating baseline models
models = [
    ("Linear Regression", linear_model.LinearRegression()),
#     ("Lasso", linear_model.Lasso()),
    ("Ridge", linear_model.Ridge()),
#     ("SDG", linear_model.SGDRegressor()),
    ("SVR", svm.LinearSVR()),
    ("NuSVR", svm.NuSVR()),
    ("SVR", svm.SVR()),
    ("Decision Tree", tree.DecisionTreeRegressor()),
    ("Random Forest", ensemble.RandomForestRegressor()),
    ("AdaBoost", ensemble.AdaBoostRegressor()),
    ("ExtraTree", ensemble.ExtraTreesRegressor()),
    ("GradientBoosting", ensemble.GradientBoostingRegressor()),
    ("K Neighbors", neighbors.KNeighborsRegressor()),
]

model_names = []
accuracies = []

# Fitting models to Training Dataset and Scoring them on Test set
for dataset_name, dataset in [("Akosomba_Data", akosombo)]:
    for model_name, model in models:
        regressor_model = model
        regressor_model.fit(X_train, y_train)

        accuracy = regressor_model.score(X_test, y_test)
        print(dataset_name, model_name, accuracy)

        model_names.append(model_name)
        accuracies.append(accuracy)


# ### Visualizing Models' Accuracy with Bar Charts

# In[7]:


# Size in inches (width, height) & resolution(DPI)
plt.figure(figsize=(34, 15), dpi=200)

x_loc = np.arange(len(models))  # the x locations for the groups
width = 0.5  # bar width

# plotting the graphs with bar chart
models_graph = plt.bar(
    x_loc, accuracies, width, color="blue", edgecolor="orange", linewidth=5,
)

plt.title("Models Accuracy", fontsize=22, pad=20)
plt.xticks(x_loc, model_names, fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.grid(b=True, which="both", axis="both", color="black", linewidth=0.8)

# adding model accuracy on top of every bar
def addLabel(models):
    for model in models:
        height = model.get_height()
        plt.text(
            model.get_x() + model.get_width() / 2.0,
            1.05 * height,
            "%f" % height,
            ha="center",
            va="bottom",
        )


addLabel(models_graph)

plt.savefig('Bar_Charts_of_Models_and_their_Accuracy.png', dpi=300, transparent=True)

plt.show()


# ### Evaluating Models

# In[8]:


# Model Evaluation
for model_name, model in models:
    helper.evaluate(X_test, y_test, model_name, model)


# ### Cross Validating Models

# #### Cross Validating with a single metric

# In[9]:


# Splitting data into 10 folds
cv_kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=23)
scorer = "r2"

model_names = []
cv_mean_scores = []
cv_std_scores = []

for model_name, model in models:
    regressor_model = model
    model_scores = model_selection.cross_val_score(
        regressor_model, X, y, cv=cv_kfold, scoring=scorer, n_jobs=-1, verbose=1,
    )
    
    print(
        f"{model_name} Accuracy: %0.2f (+/- %0.2f)"
        % (model_scores.mean(), model_scores.std() * 2)
    )

    model_names.append(model_name)
    cv_mean_scores.append(model_scores.mean())
    cv_std_scores.append(model_scores.std())


# In[10]:


cv_results = pd.DataFrame({"model_name": model_names, "mean_score": cv_mean_scores, "std_score": cv_std_scores})
cv_results.sort_values("mean_score", ascending=False, inplace=True,)
cv_results.to_csv("cross_validation_results.csv", index=True)
cv_results


# ### Visualizing Cross Validated Models with Bar Charts

# In[11]:


plt.figure(figsize=(34, 15), dpi=200)

x_loc = np.arange(len(models))
width = 0.5

models_graph = plt.bar(
    x_loc, cv_mean_scores, width, yerr=cv_std_scores, color="navy", edgecolor="orange", linewidth=5
)
plt.title("Models Cross_Validated Scores", fontsize=22, pad=20)
plt.xticks(x_loc, model_names, fontsize=20)
plt.ylabel("Accuracy", fontsize=20)
plt.grid(b=True, which="both", axis="both", color="black", linewidth=0.8)

addLabel(models_graph)

plt.savefig('Bar_Charts_of_Cross_Validated_Models_and_their_Accuracy.png', dpi=300, transparent=True)

plt.show()


# ### Training the Model with the Highest Score with Default Hyperparameters

# In[12]:


# Instantiating model object
high_score_model = ensemble.GradientBoostingRegressor()

# Fitting the model on Train set
high_score_model.fit(X_train, y_train)

# Scoring the model on Test set
high_score_model_accuracy = high_score_model.score(X_test, y_test)

print(
    f"Model without tuned hyperparameters has an accuracy of {high_score_model_accuracy}"
)


# ### Predicting with the Trained Model and Saving Predicted Results as `csv`

# In[13]:


y_pred = high_score_model.predict(X_test)

data = pd.DataFrame({"generation": list(y_test), "predicted_generation": list(y_pred),})
data.to_csv("model_predicted_values.csv", index=True)


# In[14]:


data.head(10)


# ### Optimizing the Hyperparameter of the Best Model with `GridSearchCV`

# In[15]:


# Kfold with with n_splits = 5 to split the Dataset into 5-folds
kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=23)

# Dictionary of parameters to tune
parameters = {
    "loss" : ['ls', 'lad', 'huber', 'quantile'],
    "n_estimators" : [120, 500, 800, 1200], 
    "max_depth" : [15, 25, 30, None],
    "min_samples_split" : [5, 10, 15, 100],
    "min_samples_leaf" : [1, 2, 5, 10],
    "max_features" : ["log2", "sqrt", None],
}

scorer = "r2"

# Instantiating Search object
grid = model_selection.RandomizedSearchCV(
    estimator=high_score_model, 
    param_distributions=parameters, 
    scoring=scorer, 
    cv=kfold, 
    n_jobs=-1, 
    verbose=1,
)

# Fit the grid object on Training Dataset
grid.fit(X_train, y_train)


# In[16]:


results = pd.DataFrame(grid.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
results.sort_values("rank_test_score", inplace=True)
results.to_csv("hyperparameter_tuning_results.csv", index=True)
results


# ### Evaluating the Best Estimator from the `GridSearch`

# In[17]:


best_estimator = grid.best_estimator_
helper.evaluate(X_test, y_test, "Regressor Model", best_estimator)


# ### Predicting with the Best Estimator and Saving Predicted Results as `csv`

# In[18]:


tune_y_pred = best_estimator.predict(X_test)

hyp_tune_data = pd.DataFrame(
    {"generation": list(y_test), "predicted_generation": list(tune_y_pred),}
)

hyp_tune_data.to_csv("tune_model_predicted_values.csv", index=True)

hyp_tune_data.head(10)


# ### Saving the Best Estimator with `joblib`

# In[19]:


import joblib

joblib.dump(best_estimator, "GradientBoosting.joblib")


# ### Feature Importance

# In[20]:


features = ['norminal_head', 'discharge']
importances = best_estimator.feature_importances_

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': importances,
})

feature_importance.sort_values("importance", inplace=True, ascending=False)
feature_importance.to_csv("feature_importance.csv", index=True)
feature_importance


# In[21]:


import seaborn as sns

plt.figure(figsize=(8, 5), dpi=200)

sns.barplot(x=features, y=importances, color="navy", edgecolor="orange", linewidth=5)
plt.title("Feature Importance", size=15, pad=20)
plt.xlabel("Feature", fontsize=10, labelpad=20)
plt.ylabel("Importance", fontsize=10, labelpad=20)

plt.grid(b=True, which="both", axis="both", color="black", linewidth=0.5)

plt.savefig('feature_importance.png', dpi=300, transparent=True)

plt.show()


# In[ ]:





# In[ ]:




