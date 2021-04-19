<h1 align="center">Project: Kaggle Challenge - Santander - Customer Satisfaction: 2016 (In Development)</h1>

<h1 align="center">
  <img src="https://github.com/leonvictorlima/MachineLearning_Santander_CustomerSatisfaction/blob/main/images/kagglesantander.png"  width="800"/>
</h1>

<a name="introduction"></a>

# Requirement to this project

The challenge was developed in Python 3.7.3. In order to complete and run all scripts for this project, will be necessary the use of these Python libraries:

Contents table
=================
<!--ts-->
   * [Numpy](https://numpy.org/)
   * [Pandas](https://pandas.pydata.org/)
   * [Seaborn](https://seaborn.pydata.org/index.html#)
   * [Sklearn](#collecting-data)
<!--te-->

# Project Summary

This is part of Kaggle challenge referred to identify customers satisfaction level while they are part of the Santander Bank. The main point is to evaluate their satisfaction and prevent leaving before it occurs.

The dataset and all instructions are available at: https://www.kaggle.com/c/santander-customer-satisfaction/overview

It is also necessary to have software installed to run and execute a Jupyter Notebook(http://ipython.org/notebook.html).

If you do not have python installed yet, please access Anaconda(https://anaconda.org/anaconda) website and donwload it. Anaconda offers a sort of packages integrated.

# Code

The complete template code is provided in the `Santander_CustomerSatisfaction_v1.ipynb` notebook file. It is also provided the python files which are divided into: `#Santander_CustomerSatisfaction_ExploratoryAnalysis.py`,`#[1] Santander_CustomerSatisfaction_ML_Attempt01_SMOTE_StandardScaler_GaussianNB.py`, `#[2] Santander_CustomerSatisfaction_ML_Attempt02_SMOTE_StandardScaler_LogisticRegression.py`, `#[3] Santander_CustomerSatisfaction_ML_Attempt03_StandardScaler_CrossValidation_GaussianNB.py`, `#[4] Santander_CustomerSatisfaction_ML_Attempt04_SMOTE_StandardScaler_CrossValidation_GaussianNB.py`, `#[5] Santander_CustomerSatisfaction_ML_Attempt05_StandardScaler_CrossValidation_LogisticRegression.py`, `#[6] Santander_CustomerSatisfaction_ML_Attempt06_SMOTE_StandardScaler_CrossValidation_LogisticRegression.py`, `#[7] Santander_CustomerSatisfaction_ML_Attempt07_SMOTE_StandardScaler_CrossValidation_LogisticRegression_AnotherWay.py`, `#[8] Santander_CustomerSatisfaction_ML_Attempt08_SMOTE_MaxMinScaler_PCA_CrossValidation_MachineLearningModels.py`, `#[9] Santander_CustomerSatisfaction_ML_Attempt09_StandardScaler_CrossValidation_MachineLearningModels.py`, `#[10] Santander_CustomerSatisfaction_ML_Attempt10_MaxMinScaler_PCA_CrossValidation_MachineLearningModels.py`. 

The dataset is available in `train.csv`, `test.csv`, `df3.csv` and `df4.csv` files. Please, pay attention that some codes has already been executed, maybe additional functionality will be necessary to successfully complete the project. Please fell free to enjoy and explore this project.

# Code Execution

To run the code you must navigate to `main` directory of this project and write the code below:

`jupyter notebook Santander_CustomerSatisfaction_v1.ipynb`

This must open a jupyter notebook software on your browser. 

# Data Description

You are provided with an anonymized dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers.

The task is to predict the probability that each customer in the test set is an unsatisfied customer.

Please, visit this webpage for more informations: https://www.kaggle.com/c/santander-customer-satisfaction/overview