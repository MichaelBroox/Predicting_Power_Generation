#!/usr/bin/env python
# coding: utf-8

# ### Loading Packages

# In[1]:


import pandas as pd
import numpy as np

import matplotlib as mlp
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

import bokeh as bk
from bokeh.io import output_notebook, show
output_notebook()

# custom
import helper


# ### Styling Tables

# In[2]:


get_ipython().run_cell_magic('HTML', '', "<style type='text/css'>\ntable.dataframe th, table.dataframe td{\n    border: 3px solid purple !important;\n    color: solid black !important;\n}\n</style>")


# ### Loading Dataset

# In[3]:


dataset = 'Akosombo2.xlsx'

try:
    df = pd.read_excel(dataset,'Akosombo')
    print ("Successfully loaded dataset.")
except:
    print ("Something went wrong.")

df.head()


# ### Saving Dataset Column Heads to `csv`

# In[4]:


dataset_columns = pd.DataFrame({'column_names':list(df.columns)})
dataset_columns.to_csv("column_heads_of_dataset.csv", index=True)
dataset_columns


# ### Dropping Unnecessary Columns

# In[5]:


def drop_columns(column_names, df):
    df = df.drop(column_names, axis=1)
    return df


# In[6]:


columns_to_drop = ['Date', 'Date.1', 'Upstream Elevation (feet)', 
                   'Downstream Elevation (feet)', 'Discharge (cfs)', 'Generation (W)', 
                   'Upstream Elevation (m)', 'Downstreamstream Elevation (m)', 'Efficiency', 
                   'Unnamed: 12', 'general efficiency ', 'ground acceleration']

df = drop_columns(columns_to_drop, df)

df.head()


# ### Formatting Column Heads

# In[7]:


df.columns = df.columns.str.replace(' ', '_')
df.columns = df.columns.str.lower()
df.columns


# ### Renaming Column Heads

# In[8]:


df.rename({
    'generation_(gwh)':'generation',
    'norminal_head_(h)':'norminal_head',
    'discharge_(cms)':'discharge'
}, axis='columns', inplace=True)

df.columns


# In[9]:


df.head()


# ### Checking For Missing Values and Saving Results to `csv`

# In[10]:


missing_values = helper.missing_data(df)
missing_values.to_csv("missing_values.csv", index=True)
missing_values


# In[11]:


# Saving column heads to variable
column_names = df.columns


# ### Checking For Outliers

# In[12]:


helper.detect_outliers(df)
# plt.savefig('outliers.png', dpi=300, transparent=True)


# ### Removing Outliers

# In[13]:


df = helper.outlier_remover(df)


# In[14]:


helper.detect_outliers(df)


# In[15]:


df.head()


# ### Correcting Data Index

# In[16]:


df.index = range(len(df))


# In[17]:


df = df[['norminal_head', 'discharge', 'generation']]
df.head()


# ### Checking Memory Usage

# In[18]:


df.info(memory_usage='deep')


# In[19]:


df.memory_usage(deep='True')


# ### Plotting Correlation Heat Map of Feature Variables

# In[20]:


helper.correlation_viz(df, 'correlation_of_feature_variables', save=True, dpi=300, transparent=True )


# ### Distribution of Target Variable

# In[21]:


plt.figure(figsize=(15, 10))
sns.set(color_codes=True)
sns.set_palette(sns.color_palette('muted'))

sns.distplot(df['generation'], color='maroon', bins=30)
plt.savefig('Distribution_Plot_of_PowerGeneration.png', dpi=300, transparent=True)


# ### Pairplot

# In[22]:


sns.pairplot(df, plot_kws=dict(s=40, edgecolor="orange", linewidth=1, facecolor='navy'))
plt.savefig('PairPlot.png',dpi=300, transparent=True)


# ### Histogram of Feature Variables

# In[23]:


df.hist(bins=30, figsize=(15, 10),)
plt.savefig('Histogram_of_features_variables.png', dpi=300, transparent=True)


# ### Joinplot of Feature Variables and Target Variable

# In[24]:


sns.jointplot('discharge', 'generation', df, kind='reg', height=10, ratio=10, color='maroon', )
# annot_kws=dict(stat="r"), s=40, edgecolor="orange", facecolor='navy', linewidth=1,

plt.savefig('Jointplot_of_discharge_and_target_variable.png', dpi=300, transparent=True)


# In[25]:


sns.jointplot('norminal_head', 'generation', df, kind='reg', height=10, ratio=10, color='maroon')
plt.savefig('Jointplot_of_norminal_head_and_target_variable.png', dpi=300, transparent=True)


# ### Generating Data Report

# In[26]:


import pandas_profiling as pp

profilling_report = pp.ProfileReport(df)
profilling_report.to_file('Akosombo_data_profile_report.html')


# ### Saving Data to `csv`

# In[27]:


df.to_csv('Clean_Akosombo_data.csv', index=True)


# In[ ]:




