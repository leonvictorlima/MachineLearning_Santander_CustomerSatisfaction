<h1 align="center">Kaggle Challenge - Machine Learning Santander Customer Satisfaction (In Development)</h1>

<h1 align="center">
  <img src="https://github.com/leonvictorlima/MachineLearning_Santander_CustomerSatisfaction/blob/main/images/kagglesantander.png"  width="800"/>
</h1>

<a name="introduction"></a>
# Introduction

### Kaggle Challenge - Santander - Customer Satisfaction: 2016

This is part of Kaggle challenge referred to identify customers satisfaction level while they are part of the bank. The main point is to evaluate their satisfaction and prevent leaving before it occurs.

The dataset and all instructions are available at: https://www.kaggle.com/c/santander-customer-satisfaction/overview

Data Description:
You are provided with an anonymized dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers.

The task is to predict the probability that each customer in the test set is an unsatisfied customer.


Contents table
=================
<!--ts-->
   * [Introduction](#introduction)
   * [Business problem](#business-problem)
   * [Collecting the data](#collecting-data)
   * [Exploratory Analysis](#exploratory-analysis)
   * [Feature Selection](#feature-selection)
   * [The choice of Machine Learning Model](#machine-learning)
   * [Training, Testing, and evaluate each of them](#training-test)
   * [Extra performance analysis](#perfomance-analysis)
   * [Conclusion](#conclusion)
   * [Bibliography](#bibliography)
<!--te-->

<a name="business-problem"></a>
 ## 1) Business problem:

#### Identify clients satisfaction levels before them leave:
    0 = satisfied
    1 = insatisfied

<a name="collecting-data"></a>    
## 2) Collecting data:

  The dataset is provided by Kaggle's webpage.
  
```python
from pandas import read_csv

df = read_csv("C:/train.csv", sep = ',')
```
<a name="exploratory-analysis"></a> 
## 3) Exploratory Analysis:

As part of any project, the analysis of dataset must be executed in order to evaluate and getting familiar with data inserted in it.

```python
# import libraries

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Head of dataset

df.head()

# dataset's dimmension 
df.shape

df.dtypes

# NA values?

df.isnull().values.any()
```

### We must have in mind, any changes made in training dataset have to be applied to test dataset.

```python
# In the dataset description provided by Kaggle is that only numeric variables are available, we must have sure tough.
# Then, Exist strings?

df.select_dtypes(include=[object]).sum()

[df[items] for items in df if '' in df[items]]

# Describe values

df.describe()
```
<h1 align="center">
  <img src="https://github.com/leonvictorlima/MachineLearning_Santander_CustomerSatisfaction/blob/main/images/describe2.JPG"  width="800"/>
</h1>

As the dataset is too big, it is hard to find out all issues. For that reason, it is important at this moment to pay attention at Target , Age (var15) variables and realize that all values are not in the same scale.

For the Target value, we can see the number of results are different for each one. So, whether use the data this way predictions problems will appear.

```python
# Grouping dataset by Target variable;

df.groupby('TARGET').size()

# Observing var15 (age) variable with more attention. 

df.var15.describe()

df.var15.value_counts()
```
<h1 align="center">
  <img src="images/target_size.JPG"  width="800"/>
</h1>

### Plots

Creating plots is the best way to understanding data behaviors.

```python
# Thinking about build plots, we must have one variables list to do it.

columns_list = list(df.columns)

# taking out "TARGET" and "ID" variables

columns_list.pop(-1) # TARGET
columns_list.pop(0) # ID
```

Using set of plots below, it is possible realize that variables wich begins with "delta" or "ind" have two values: 0 or 1. On the other hand, several variables showed has outliers as well as are not in normal distribuition.

```python

for items in columns_list:
    fig, ax = plt.subplots(1,2, figsize=(8, 4))
    sns.boxplot(df[items], orient = 'v', ax=ax[0])
    sns.distplot(df[items], ax=ax[1])
    fig.tight_layout()  
```
<h1 align="center">
  <img src="images/variables_plot.JPG"  width="800"/>
</h1>

### Correlation between variables 

It is crucial for data projects to determine the correlation between its features. Following this dataset and its enormous numbers of variables, for that reason, estimate and dive deeply into its variables correlation with Target variable is a fundamental matter to reduce dimensionality and get to how they are connected.

```python

# Creating correlation 

corr=df.corr()

# Analyzing its correlations

corr.TARGET.sort_values(ascending = False)
```
