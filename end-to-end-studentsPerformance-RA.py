#!/usr/bin/env python
# coding: utf-8

# # End-to-end Machine Learning project
# 
# **Original Dataset Source: Cortez,Paulo. (2014). Student Performance. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.**
# 
# 
# **Modified Dataset: Student Performance https://raw.githubusercontent.com/rubaomari/data/main/student/student-mat-modified-RA.csv**

# # Student Performance Dataset Description
# 
# **Attributes for student-mat-modified-RA.csv dataset:**
# The below attributes are copied **AS IS** from the original dataset.
# 
# 1. school - student's school (binary: "GP" - Gabriel Pereira or "MS" - Mousinho da Silveira)
# 
# 2. sex - student's sex (binary: "F" - female or "M" - male)
# 
# 3. age - student's age (numeric: from 15 to 22)
# 
# 4. address - student's home address type (binary: "U" - urban or "R" - rural)
# 
# 5. famsize - family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
# 
# 6. Pstatus - parent's cohabitation status (binary: "T" - living together or "A" - apart)
# 
# 7. Medu - mother's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 
# 8. Fedu - father's education (numeric: 0 - none,  1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# 
# 9. Mjob - mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# 
# 10. Fjob - father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
# 
# 11. reason - reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
# 
# 12. guardian - student's guardian (nominal: "mother", "father" or "other")
# 
# 13. traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# 
# 14. studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# 
# 15. failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# 
# 16. schoolsup - extra educational support (binary: yes or no)
# 
# 17. famsup - family educational support (binary: yes or no)
# 
# 18. paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# 
# 19. activities - extra-curricular activities (binary: yes or no)
# 
# 20. nursery - attended nursery school (binary: yes or no)
# 
# 21. higher - wants to take higher education (binary: yes or no)
# 
# 22. internet - Internet access at home (binary: yes or no)
# 
# 23. romantic - with a romantic relationship (binary: yes or no)
# 
# 24. famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# 
# 25. freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# 
# 26. goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# 
# 27. Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# 28. Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# 
# 29. health - current health status (numeric: from 1 - very bad to 5 - very good)
# 
# 30. absences - number of school absences (numeric: from 0 to 93)
# 
# 31. G1 - first period grade (numeric: from 0 to 20)
# 
# 32. G2 - second period grade (numeric: from 0 to 20)
# 
# 33. G3 - final grade (numeric: from 0 to 20, output target)
# 
# 
# **Missing values:** The original dataset in UCI doesn't have any missing values. I have randomly removed values so we can discuss what to do with missing data.
# 
# The following number of attributes have been removed from the original dataset:
# - Sex : 3
# - Age: 3
# - Family size: 7
# - guardian: 262
# - Study time: 3
# 
# 
# **Duplicated values:** 
# 9 duplicated values have been created

# # 1- Look at the big picture and frame the problem.

# ### Frame the problem
# 1. Supervised learning – training examples are labeled.
# 2. A regression task – predict a value.
# 3. Batch learning 
#     - Small data set
#     - No continuous flow of data coming into the system
#     - No need to adjust to changing data rapidly
# 
# ### Look at the big picture
# Predictions will be used to help inform student advisors of students at risks. Advisors will use the info to contact students to discuss thier academic standing and let them know of the available support.

# In[2]:


# Import libraries

import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import certifi
import ssl

os.environ['SSL_CERT_FILE'] = '/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/certifi/cacert.pem'

# Override the default SSL context with a custom context using the specified cafile
#custom_context = ssl.create_default_context(cafile=custom_ca_file)

# Now you can use this custom context for your requests
import requests


# # 2- Load the dataset
# 
# Open the dataset using Pandas and load it into a DataFrame, which is the object Pandas uses to store tables of data.
# Pandas uses two objects for storing datasets: the DataFrame and the Series. 
# Series is used for datasets with only one column, and DataFrame is used for datasets of more than one column.

# In[3]:


# Load the dataset
# If there is no separator, you can use df = pd.read_csv(url)
# If the dataset is on your local machine use df = pd.read_csv("datasets/student/student-mat-modified-RA.csv")
# If reading from UCI directly, some datasets are available thorugh the ucimlrepo (https://github.com/uci-ml-repo/ucimlrepo)

url = "https://raw.githubusercontent.com/rubaomari/data/main/student/student-mat-modified-RA.csv"
students = pd.read_csv(url, sep=',')

# Let's create a backup copy of the dataset
studnets_backup = students


# ## 2.1 Take a quick look at the data structure

# Examine the number of rows, the number of columns, and columns labels in the dataset.
# 
# - When you use the pd.read_csv() function without specifying any arguments, Pandas assumes that the first row of the CSV file contains column headers.
# 
# - If your CSV file does not contain headers, you can explicitly specify this by setting the header argument to None. For example: df = pd.read_csv('data.csv', header=None)
# 
# - If your CSV file contains headers but they are not in the first row, you can specify which row to use as headers using the header argument. For example, if your header is in the second row: df = pd.read_csv('data.csv', header=1) 
# 
# - You can also provide your own custom column names using the names argument. This is useful if your CSV file doesn't have headers, and you want to provide meaningful column names. For example:
# 
#     custom_headers = ["A", "B", "C"]
#     
#     df = pd.read_csv('data.csv', header=None, names=custom_headers)

# In[4]:


students


# 
# #### Use head() to look at the first 5 rows, you can also choose to use another number, e.g., head(10).
# 

# In[5]:


students.head()


# #### Use describe() method to see a summary of the numerical attributes.
# 
# 
# Example: 25% of the students have an average age lower than 16, while 50% are lower than 17 and 75% are lower than 16.

# In[6]:


students.describe()


# #### Use info() to get a quick description of the data, the total number of rows, each attribute’s type, and the number of non-null values.
# 
# #### The original dataset in UCI doesn't have any missing values. I have randomly removed values so we can discuss what to do with missing data.

# In[7]:


students.info()


# #### For categorical data, use value_counts to show what categories exist and how many rows belong to each category

# In[8]:


students["school"].value_counts()


# ## 2.2 Create a Test Set

# Typically, you want to create the test set here, then explore and visualise the training data only to gain insights.
# You will also typically prepare the data preprcosssing pipeline using training data only, and then apply the preprocessing on the test dataset.
# However, we will do the data cleaning and preprocessing on the whole dataset, then split it into training and testing datasets right before training the model.
# 

# # 3. Explore and visualize the data to gain insights.
# 

# ### 3.1 Plot a histogram of the data using hist()
# 
# The towers or bars of a histogram are called bins. The height of each bin shows how many values from that data fall into that range. 
# 
# Width of each bin is = (max value of data – min value of data) / total number of bins
#     - age bin width = (22-15)/20 = 0.35
# 
# The default value of the number of bins to be created in a histogram is 10. However, we can change the size of bins using the parameter bins

# In[9]:


students.hist(figsize=(24, 16))
plt.show()


# #### Note that these attributes have very different scales. We will scale these attributes later.

# ## 3.2 Look for correlations between the features

# In[10]:


# Check for correlation between attributes using sns.pairplot.
sns.pairplot(students)


# ### Look for correlations using Pearson correlation coefficient. Plot G1 vs. G3 using sns.lineplot, and create a scatterplot for G2 and G3.

# In[11]:


#corr method has pearson standard correlation coefficient as the default 

corr_matrix = students.corr(numeric_only=True)
corr_matrix


# #### Let's look at correlations with regard to our target

# In[12]:


corr_matrix["G3"].sort_values(ascending=False)


# In[13]:


# Plot G1 vs. G3

G1vsG3 = sns.lineplot(x="G1", y="G3", data=students, errorbar=None)


# In[14]:


# Create a scatterplot of G2 and G3

X = students["G2"]
y = students["G3"]
# Plot points
fig, pl = plt.subplots()
pl.scatter(X, y, color = 'b')
plt.xlabel("G2")
plt.ylabel("G3")


# # 4. Prepare the data for Machine Learning Algorithms

# #### Let's start by cleaning up the data: duplicate and missing values and how to deal with them

# ### 4.1-Check for duplicate rows and remove them if any.

# In[15]:


# Check for duplicate rows and delete them

students.duplicated().sum()


# In[16]:


students.drop_duplicates()


# ### 4.2-Handle the missing values

# In[17]:


# Find the number of missing values in each column

students.isna().sum()


# 
# Note for many online datasets, you will find missing values are presented by "?".
# 
# You can replace the "?" with NaN first then follow the same steps above.
# 
# (df.values == '?').sum()
# 
# df = df.replace('?', np.nan)
# 
# df.isna().sum()

# #### For the guardian feature, we will delete the whole feature
#     
# When dropping a column, we use the inplace=True option which is an argument in many pandas DataFrame methods that allows you to modify the object directly without having to create a new copy. 
# 
# When you set inplace=True, the operation will be applied to the object itself, and it will modify the original object.
# 
# This can be useful for saving memory and improving performance when working with large datasets, as it avoids creating unnecessary copies of the data. However, it can make your code harder to debug.
# 

# In[18]:


students.drop(labels=['guardian'], axis=1, inplace=True)


# The axis parameter specifies whether you want to drop rows (axis=0) or columns (axis=1).

# In[19]:


students.info()


# In[20]:


students.isna().sum()


# #### For the famsize attribute we will delete the examples

# In[21]:


students.dropna(subset=["famsize"], inplace=True)


# In[22]:


# Note the first three examples are deleted

students


# In[23]:


students.isna().sum()


# #### For the remaining missing values, we will fill them with the mean if it is a numerical value and the most frequent if it is a categorical column
# 
# #### We will do this through creating a pipeline, that will also scale the features and perform encoding in the next step.
# 
# 
# 

# ### 4.3. Create a pipeline that will:
# 
# 1. Fill in the missing numerical values with the mean using a SimpleImputer
# 
# 2. Scale the numerical columns using StandardScaler. Do not scale the target
# 
# 3. Fill in the missing categorical values with the most_frequent value using SimpleImputer
# 
# 4. Encode the categorical columns using OneHotEncoder
# 

# In[24]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# In[25]:


# Create the cat and num columns
# Get a list of column names from the 'students' DataFrame that are of numerical data types.
# Get a list of column names from the 'students' DataFrame that are not of numerical data types.

num_cols = students.select_dtypes(include='number').columns.to_list()
cat_cols = students.select_dtypes(exclude='number').columns.to_list()


# Exclude the target from numerical columns
num_cols.remove("G3")

# Create pipelines for numeric and categorical columns
num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())

# Use ColumnTransformer to set the estimators and transformations

preprocessing = ColumnTransformer([('num', num_pipeline, num_cols),
                                   ('cat', cat_pipeline, cat_cols)],
                                    remainder='passthrough'
                                 )


# In[26]:


num_cols


# In[27]:


cat_cols


# #### Display your pipeline

# In[28]:


# Show the pipeline
preprocessing


# In[29]:


# Apply the preprocessing pipeline on the dataset

students_prepared = preprocessing.fit_transform(students)

# Scikit-learn strips the column headers, so just add them back on afterward.
feature_names=preprocessing.get_feature_names_out()
students_prepared = pd.DataFrame(data=students_prepared, columns=feature_names)

students_prepared


# In[30]:


students_prepared.columns


# # 5. Select a model and train it

# ## 5.1 Split the dataset into a training dataset (80%) and testing dataset.

# In[31]:


from sklearn.model_selection import train_test_split

X = students_prepared.drop(["remainder__G3"], axis=1)
y = students_prepared["remainder__G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# ### 5.2 Train a Linear Regression model with no regularization

# In[32]:


from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

lr_model.fit(X_train,y_train)


# ### Test your model on the test set, and report on the Mean Squared Error

# In[33]:


# Predict the outcome of test data

lr_y_predict = lr_model.predict(X_test)

from sklearn.metrics import mean_squared_error as mse
lr_mse=mse(y_test, lr_y_predict)
lr_mse


# ### 5.3 Train a Linear Regression model using KFold cross-validation with 5 folds, and report on the cross validation score, use negative mean squared error as the cross validation metric.

# In[34]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

scores


# ### 5.4 Calculate the mean and standard deviation of the cross-validation scores to get an overall assessment of the model's performance:

# In[35]:


mean_score = -scores.mean()  # Take the negative value to get the mean squared error

print(f'Cross-Validation Mean Score: {mean_score}')


# FYI, cross_val_score in scikit-learn does allow you to use mean squared error (MSE) as a scoring metric. 
# 
# However, you should use 'neg_mean_squared_error' as the scoring parameter when you call cross_val_score. 
# 
# The reason for this naming convention is that cross_val_score follows a convention where greater values are better for scoring, while MSE is typically minimized (you want it to be as small as possible). 
# 
# To reconcile this, you use the negative MSE, which turns it into a maximizing score (higher is better).
# 
# Here's how you can use cross_val_score with negative mean squared error:
# 
# scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(mean_squared_error, greater_is_better=False))
# 
# 

# ### 5.5 Train a Linear Regression model using Ridge and Lasso with alpha=1

# In[36]:


from sklearn.linear_model import Ridge, Lasso

RidgeRegression = Ridge(alpha=1)
ridge_model = RidgeRegression.fit(X_train, y_train)


LassoRegression = Lasso(alpha=1)
lasso_model = LassoRegression.fit(X_train, y_train)


# ### Test your models on the test dataset and report on the MSE

# In[37]:


Ridge_y_predict = ridge_model.predict(X_test)
Lasso_y_predict = lasso_model.predict(X_test)
ridge_mse = mse(y_test, Ridge_y_predict)
lasso_mse=mse(y_test, Lasso_y_predict)

print(f'Ridge Regression MSE: {ridge_mse}')
print(f'Lasso Regression MSE: {lasso_mse}')


# Compare the results from all the models

# In[38]:


print(f'Linear Regression MSE: {lr_mse}')
print(f'Cross-Validation Mean Score: {mean_score}')
print(f'Ridge Regression MSE: {ridge_mse}')
print(f'Lasso Regression MSE: {lasso_mse}')


# ### 5.6 Plot the prediction vs. actual for the best performing model

# In[39]:


import matplotlib.pyplot as plt

# Assuming 'model' is your trained linear regression model
lr_y_predict = lr_model.predict(X_test)  # X is your feature data
plt.scatter(Ridge_y_predict, y_test)  # y is your actual target values
plt.xlabel("G3 Predicted Values")
plt.ylabel("G3 Actual Values")
plt.title("Predicted vs. Actual Values")
plt.show()


# # Markdown for Jupyter notebooks cheatsheet
# 
# Source: https://www.ibm.com/docs/en/watson-studio-local/1.2.3?topic=notebooks-markdown-jupyter-cheatsheet
# 
# Here's how to format Markdown cells in Jupyter notebooks:
# 
# Headings
# 
# Use the number sign (#) followed by a blank space for notebook titles and section headings:
# # for titles
# ## for major headings
# ### for subheadings
# #### for 4th level subheadings
# 
# ### Emphasis
# 
# Use the following code to emphasize text:
# 
# Bold text: __string__ or **string**
# 
# Italic text: _string_ or *string*
# 
# 
# ### Bullets
# 
# To create a circular bullet point, use one of the following methods. Each bullet point must be on its own line.
# 
# A hyphen (-) followed by one or two spaces, for example: - Bulleted item
# 
# A space, a hyphen (-) and a space, for example: - Bulleted item
# 
# An asterisk (*) followed by one or two spaces, for example: * Bulleted item
# 
# ## Numbered lists
# 
# To create a numbered list, enter 1. followed by a space, for example:
# 1. Numbered item
# 1. Numbered item
# 
# For simplicity, you use 1. before each entry. The list will be numbered correctly when you run the cell.
# 
# 

# In[ ]:




