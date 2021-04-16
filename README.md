<h1 align="center">Kaggle Challenge - Machine Learning Santander Customer Satisfaction (In Development)</h1>

<h1 align="center">
  <img src="https://github.com/leonvictorlima/Machine-Learning-Credit-Analysis-Data-Science/blob/main/Images/credit.jpg"  width="500"/>
</h1>

<a name="introduction"></a>
# Introduction

### Kaggle Challenge

Related to Kaggle analysis for Santander Bank satisfaction client;

### Santander - Customer Satisfaction: 2016

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
df = read_csv("C:/train.csv", sep = ',')
```
<a name="exploratory-analysis"></a> 
## 3) Exploratory Analysis:

As part of any project the analysis of dataset must be executed in order to evaluate and getting familiar with data inserted in it.

