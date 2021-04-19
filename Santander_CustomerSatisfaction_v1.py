# Declare libraries

from pandas import read_csv

df = read_csv("C:/FCD/BigDataAnalyticsPythonSpark/Projeto03/train.csv", sep = ',')

# Exploratory Analysis

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Head of dataset

df.head()

# shape
df.shape

# variable types
df.dtypes

### We must have in mind, any changes made in training dataset have to be applied to test dataset.

# In the dataset description provided by Kaggle is that only numeric variables are available, we must have sure tough.
# Then, Exist strings?

df.select_dtypes(include=[object]).sum()

[df[items] for items in df if '' in df[items]]

# Describe values

df.describe()

#As the dataset is too big, it is hard to find out all issues. For that reason, it is important at this moment to pay attention at Target , #Age (var15) variables and realize that all values are not in the same scale.

#For the Target value, we can see the number of results are different for each one. So, whether use the data this way predictions problems #will appear.

# Grouping dataset by Target variable;

df.groupby('TARGET').size()

# Observing var15 (age) variable with more attention. 

df.var15.describe()

# Value counts

df.groupby("var15").size() #by size

df.var15.value_counts() #by value_counts ordered

### Plots

#Creating plots is the best way to understanding data behaviors.

# help to know more abou this function

?plt.hist

# Seaborn version

sns.__version__

# Test Seaborn plot

sns.distplot(df.var15)

# Test Seaborn boxplot

sns.boxplot(df.var15, orient = 'v')

# Identifing dataframe columns

df.columns

# Let's create a new one...

sns.boxplot(df.imp_ent_var16_ult1, orient='v')

# Thinking about build plots, we must have one variables list to do it.

columns_list = list(df.columns)
columns_list

# taking out "TARGET" and "ID" variables

columns_list.pop(-1) 
columns_list.pop(0)
columns_list

# Using set of plots below, it is possible realize that variables wich begins with "delta" or "ind" have two values: 0 or 1. 
# On the other hand, several variables showed has outliers as well as are not in normal distribuition.

for items in columns_list:
    fig, ax = plt.subplots(1,2, figsize=(8, 4))
    sns.boxplot(df[items], orient = 'v', ax=ax[0])
    sns.distplot(df[items], ax=ax[1])
    fig.tight_layout()


# Evaluating skewness and Kurtosis. 

# Skewness is to analyze the shape of each variable in relation to x axis variance, while kurtosis is related to y axis 
# shape variance

# coenf = 0 -- There is no assimetry
# coef > 1 -- Positive assimetry - data more to left (skewness) / High (kurtosis)
# coef < 1 -- Negative assimetry - data more to right (skewness) / low (kurtosis)


df.skew() # Skewness

df.kurt() # Kurtosis

### Correlation between variables 

# It is crucial for data projects to determine the correlation between its features. 
# Following this dataset and its enormous numbers of variables, for that reason, estimate and dive deeply into its variables correlation
# with Target variable is a fundamental matter to reduce dimensionality and get to how they are connected.

# Creating correlation 

corr=df.corr()

# Analyzing its correlations

corr.TARGET.sort_values(ascending = False)

# As seen before, exist some variables which not present any relationship with target variable. 
# Here they are listed:Therefore, its values will be excluded of variables list for next steps.

# List of variables which have not relationship with target;

variables_out = corr[corr['TARGET'].isnull()].index.tolist
variables_out

# One important choice for any data science project is selection of good variables, normalizing data, and feature selection as well. 
# In addition, data normalization is a contunded method to put all features set in the same scale. By the way, it is important keep in mind
# the type of feature selection will be made and machine learning model will be implemented in next steps as well.
# However, before get started into this, lets create a range of age because the current range can negativetly to contribute with our 
# machine learning model.

## Changing Range of age by 

df.var15.unique()

df.var15.value_counts().index

# Learning about plt.bar
?plt.bar

# Plotting
plt.bar(df.var15.value_counts().index, df.var15.value_counts())

# The main point here is to group the var15 variable in ranges. Following that, is to use LabelEnconder technique which this 
# new range data will be supersede by number. In the end, we are going to use OneHotEnconder to catalog this ranges in arrays with each
# element in the range signed for its respective value adding all new variables age on dataframe.

df2 = df.copy()
df2.head()
pd.cut(x = df['var15'], bins = bins, right = True)
df2.insert(loc = 3, column = 'var15_range', value = pd.cut(x = df['var15'], bins = bins, right = True))
df2[df2['var15_range'].isnull()].index.tolist()
df2.isnull().values.any()

# Machine Learning does not work with categorical values. Moreover, variables wich are part of any modelation must have data integrity.
# As part of data muging process was identified some contunded data to be implemented and make use on this approach. 
# In addition, transformation is perfomed to modify and handle with its challenges. 

# Concluded the range step, now we are going to introduce LabelEnconder technique, because var15 grouped by ranges
# the data has changed to categorical type and we must convert it to numerical type again.

# LabelEncoder technique will insert differents values to each range data created before. With that technique, 
# the variable evaluated return to numeric type. By the way, the score given to each item can to have one contrary 
# benefaction in its current form, and to handle it another technique must be executed named One Hot Encoder. 

df2.var15_range.dtypes

## Loading library to LabelEnconder

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# Applyin encoder wich returns an array
labels_var15_range = label_encoder.fit_transform((df2.var15_range).astype('str'))
len(labels_var15_range)

df2.insert(loc = 4, column = 'Labels_var15_range', value = labels_var15_range)
df2.head(20)

# Thus far, we converted all elements in var15 to a range of values and afterwards modified it, maneuvering with LabelEncoder. 
# As mentioned previously, data post labelEncoder can affect the balance when applied a machine learning algorithm due to weight 
# given to the data range. On this line, it is considered put in practice another technique named One Hot Encodering.

# One Hot Encoder will result in one array with equals size for the range and designating 1 or 0 for its due property. 
# Sequently, the next step is transforming its new variables for one dataset and concatenate all elements in new one. 

# Applying OneHotEncoder

from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()

var15_array = one_hot_encoder.fit_transform(df2[['Labels_var15_range']]).toarray()
var15_array

# Taking labels 

label_encoder.classes_

# Converting array to dataframe

var15_df=pd.DataFrame(data = var15_array, columns = 'var15_'+label_encoder.classes_)
var15_df.head()

# Creating a new consolidated dataframe implementing a new function: df3

def insert_dataframe_position(position, dataframe_1, dataframe_2):
    dataframe_3=dataframe_1.copy()
    for items in range(dataframe_2.shape[1]):
        dataframe_3.insert(loc=position, column=dataframe_2.columns[items], value=dataframe_2.iloc[:,items])
        position=position+1
    return dataframe_3

df3=insert_dataframe_position(5,df2,var15_df)
df3.head(10)

# Saving our progress made up now!

df3.to_csv('df3.csv', index_column = False)

# Before begin with normalizing method, variables which are not part of our method needs to be taken out.
# One example of unnecessary features starts with ID, var15, var15_range, Labels_var15_range, and on and on.

# In addition, the list contain variables without any relationship or correlation with TARGET value.         