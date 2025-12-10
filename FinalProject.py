#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
    classification_report
)

# for stepwise selection later
from sklearn.linear_model import LogisticRegression


# In[45]:


# Read in the data, call the dataframe "s"  and check the dimensions of the dataframe
s = pd.read_csv('social_media_usage.csv')


# In[46]:


s.shape


# In[47]:


# Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1.
# If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and 
# two columns and test your function to make sure it works as expected

def clean_sm(x): 
    return np.where(x==1,1,0)


# In[48]:


# toy dataframe with 3 rows and 2 columns
r = pd.DataFrame({
    'col1': [1, 0, 2],
    'col2': [1, 3, 1]
})

r_clean = r.apply(clean_sm)
print(r_clean)


# Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[49]:


ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]), # target
    "income": np.where(s["income"].between(1, 9), s["income"], np.nan), #using income
    "education": np.where(s["educ2"].between(1, 8), s["educ2"], np.nan), # using education
    "parent": clean_sm(s["par"]), # binary               
    "married": clean_sm(s["marital"]), # binary  
    "female": clean_sm(s["gender"]), # binary  
    "age": np.where(s["age"] <= 98, s["age"], np.nan) # age above 98 NA:
})

# drop missing values
ss = ss.dropna()

print("Cleaned dataframe ss:")
print(ss.head())
print("\nShape:", ss.shape)


# In[50]:


# exploratory analysis
ss.describe()


# In[51]:


# more exploratory analysis
for col in ["parent", "married", "female"]:
    print(f"\nMean LinkedIn use by {col}:")
    print(ss.groupby(col)["sm_li"].mean())

print("\nCorrelation between age and LinkedIn use:")
print(ss["age"].corr(ss["sm_li"]))


# In[52]:


# plots for exploratory analysis

sns.boxplot(x="sm_li", y="age", data=ss)
plt.title("Age distribution by LinkedIn use")
plt.show()

sns.barplot(x="income", y="sm_li", data=ss)
plt.title("LinkedIn use rate by income")
plt.show()

sns.barplot(x="education", y="sm_li", data=ss)
plt.title("LinkedIn use rate by education")
plt.show()


# In[53]:


# Create a target vector (y) and feature set (X)

y = ss["sm_li"]
X = ss[["income","education","parent","married","female","age"]]


# In[55]:


# Split the data into training and test sets. Hold out 20% of the data for testing. 
# Explain what each new object contains and how it is used in machine learning

import statsmodels.api as sm
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=8675309) # set for reproducibility

# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.


# In[59]:


# Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# Train logistic regression
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)


# In[65]:


# Evaluate the model using the testing data. What is the model accuracy for the model? 
# Predict on ORIGINAL test data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy:", accuracy)
print ("The model correctly predicted LinkedIn use for 66% of users")

# Use the model to make predictions and then generate a confusion matrix from the model. 
confusion_matrix(y_test, y_pred)

# Interpret the confusion matrix and explain what each number means.
# 104 (top left): true negative (TN) values: 104 people were correctly predicted as not using LinkedIn (0)
# 64 (top right): false positive (FP) values: 64 people who do not use LinkedIn were predicted as users (1)
# 22 (bottom left): false negative (FN) 22 people who do use LinkedIn were predicted to not use (0)
# 62 (top left): true positive (TP) 62 people who use LinkedIn were correctly predicted as users (1)


# In[66]:


# Create the confusion matrix as a dataframe and add informative column names and index names 
# that indicate what each quadrant represents
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")


# In[75]:


# Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 
# Use the results in the confusion matrix to calculate each of these metrics by hand. 
# Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. 

# Precision: TP/(TP+FP), good to use when goal is to minimize incorrectly predicting positive cases
# Example: spam detection, do not want to mark real emails as spam
precision = 62/(62+64)
print("Precision:",precision)

# Recall: TP/(TP+FN), important when the goal is to minimize the chance of missing positive cases
# Example: cancer detection, do not want to miss positive cancer case
recall = 62/(62+22)
print("Recall:",recall)

# F1: 2 * ((precision*recall)/(precision+recall)), good when want to minimize both the chance of missing positive cases
# and to minimize incorrectly predicting positive cases, good also for imbalanced data
# Example: fraud detection, catch fraud but do not over catch/annoy innocent people
f1 = 2 * ((precision*recall)/(precision+recall))
print("F1:",f1)

# After calculating the metrics by hand, create a classification_report using sklearn 
# and check to ensure your metrics match those of the classification_report.
print(classification_report(y_test, y_pred))
print("After evaluation, hand and computed calculations match")


# In[92]:


# Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8),
# with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? 

# Prediction 1: from list income, education, parent, married, female, age
person = [8,7,0,1,1,42]

# Predict user class, given input features (0 for not a user, 1 for is a user)
predicted_class = model.predict([person])

# Generate probability of positive class (=1)
probs = model.predict_proba([person])


# In[95]:


# Print predicted class and probability
print(f"Predicted Class: {predicted_class[0]}") # 0=not a user, 1=is a Linkedin User
print(f"Probability that this person is a Linkedin User: {probs[0][1]}")


# In[96]:


# How does the probability change if another person is 82 years old, but otherwise the same?

# Prediction 2: from list income, education, parent, married, female, age
person = [8,7,0,1,1,82]

# Predict user class, given input features (0 for not a user, 1 for is a user)
predicted_class = model.predict([person])

# Generate probability of positive class (=1)
probs = model.predict_proba([person])


# In[97]:


# Print predicted class and probability
print(f"Predicted Class: {predicted_class[0]}") # 0=not a user, 1=is a Linkedin User
print(f"Probability that this person is a Linkedin User: {probs[0][1]}")

