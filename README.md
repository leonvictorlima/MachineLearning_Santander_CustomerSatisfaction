<h1 align="center">Project: Kaggle Challenge - Santander - Customer Satisfaction: 2016 (In Development)</h1>

<h1 align="center">
  <img src="https://github.com/leonvictorlima/MachineLearning_Santander_CustomerSatisfaction/blob/main/images/kagglesantander.png"  width="800"/>
</h1>

<a name="introduction"></a>

# Classification Learning

## Project Requirements

The challenge was developed in Python 3.7.3. In order to complete and run all scripts for this project, will be necessary the use of these Python libraries:

## Python Libraries
=================
<!--ts-->
   * [Numpy](https://numpy.org/)
   * [Pandas](https://pandas.pydata.org/)
   * [Seaborn](https://seaborn.pydata.org/index.html#)
   * [imblearn](http://glemaitre.github.io/imbalanced-learn/generated/imblearn.over_sampling.SMOTE.html)
   * [Sklearn](https://scikit-learn.org/)
<!--te-->

## Project Summary

This is part of Kaggle challenge referred to identify customers satisfaction level while they are part of the Santander Bank. The main point is to evaluate their satisfaction and prevent leaving before it occurs.

The dataset and all instructions are available at: https://www.kaggle.com/c/santander-customer-satisfaction/overview

It is also necessary to have software installed to run and execute a Jupyter Notebook(http://ipython.org/notebook.html).

If you do not have python installed yet, please access Anaconda(https://anaconda.org/anaconda) website and donwload it. Anaconda offers a sort of packages integrated.

## Code

The complete template code is provided in the `Santander_CustomerSatisfaction_v1.ipynb` notebook file. It is also provided the python files which are divided into: `#Santander_CustomerSatisfaction_ExploratoryAnalysis.py`,`#[1] Santander_CustomerSatisfaction_ML_Attempt01_SMOTE_StandardScaler_GaussianNB.py`, `#[2] Santander_CustomerSatisfaction_ML_Attempt02_SMOTE_StandardScaler_LogisticRegression.py`, `#[3] Santander_CustomerSatisfaction_ML_Attempt03_StandardScaler_CrossValidation_GaussianNB.py`, `#[4] Santander_CustomerSatisfaction_ML_Attempt04_SMOTE_StandardScaler_CrossValidation_GaussianNB.py`, `#[5] Santander_CustomerSatisfaction_ML_Attempt05_StandardScaler_CrossValidation_LogisticRegression.py`, `#[6] Santander_CustomerSatisfaction_ML_Attempt06_SMOTE_StandardScaler_CrossValidation_LogisticRegression.py`, `#[7] Santander_CustomerSatisfaction_ML_Attempt07_SMOTE_StandardScaler_CrossValidation_LogisticRegression_AnotherWay.py`, `#[8] Santander_CustomerSatisfaction_ML_Attempt08_SMOTE_MaxMinScaler_PCA_CrossValidation_MachineLearningModels.py`, `#[9] Santander_CustomerSatisfaction_ML_Attempt09_StandardScaler_CrossValidation_MachineLearningModels.py`, `#[10] Santander_CustomerSatisfaction_ML_Attempt10_MaxMinScaler_PCA_CrossValidation_MachineLearningModels.py`. 

The dataset is available in `train.csv`, `test.csv`, `df3.csv` and `df4.csv` files. Please, pay attention that some codes has already been executed, maybe additional functionality will be necessary to successfully complete the project. Please feel free to enjoy and explore this project.

## Code Execution

To run the code you must navigate to `main` directory of this project and write the code below:

`jupyter notebook Santander_CustomerSatisfaction_v1.ipynb`

This must open a jupyter notebook software on your browser. 

## Data Description

You are provided with an anonymized dataset containing a large number of numeric variables. The "TARGET" column is the variable to predict. It equals one for unsatisfied customers and 0 for satisfied customers.

The task is to predict the probability that each customer in the test set is an unsatisfied customer.

Please, visit this webpage for more informations: https://www.kaggle.com/c/santander-customer-satisfaction/overview

## Bibliography


1. Kaggle Santander Challenge: https://www.kaggle.com/c/santander-customer-satisfaction/
2. Data Science Academy: https://www.datascienceacademy.com.br/
3. Unbalanced Datasets & What To Do About Them: https://medium.com/strands-tech-corner/unbalanced-datasets-what-to-do-144e0552d9cd
4. Imbalanced-learn documentation: https://imbalanced-learn.org/stable/
5. Como lidar com dados desbalanceados em problemas de classificação: https://medium.com/data-hackers/como-lidar-com-dados-desbalanceados-em-problemas-de-classifica%C3%A7%C3%A3o-17c4d4357ef9
6. How to handle imbalanced classes in Machine Learning: https://elitedatascience.com/imbalanced-classes
7. How to Deal with Imbalanced Data using SMOTE: https://medium.com/analytics-vidhya/balance-your-data-using-smote-98e4d79fcddb
8. 5 SMOTE Techniques for Oversampling your Imbalance Data: https://towardsdatascience.com/5-smote-techniques-for-oversampling-your-imbalance-data-b8155bdbe2b5
9. Sklearn documentation: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
10. Sklearn Pipeline documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html
11. Comparative Analysis of Oversampling Techniques on Imbalanced Data: https://towardsdatascience.com/comparative-analysis-of-oversampling-techniques-on-imbalanced-data-cd46f172d49d
12. vanisinghal0201 github portfolio: https://github.com/vanisinghal0201/Comparative_Analysis/blob/master/SamplegenerationUsingSMOTE.ipynb
13. Learning from imbalanced data: https://www.jeremyjordan.me/imbalanced-data/
14. imblearn documentation: http://glemaitre.github.io/imbalanced-learn/generated/imblearn.over_sampling.SMOTE.html
15. Scale, Standardize, or Normalize with Scikit-Learn: https://towardsdatascience.com/scale-standardize-or-normalize-with-scikit-learn-6ccc7d176a02
16. Sklearn Gaussian NB documentation: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
17. Sklearn GridSearchCV documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
18. SKlearn: Pipeline & GridSearchCV: https://medium.com/@cmukesh8688/sklearn-pipeline-gridsearchcv-54f5552bbf4e#:~:text=Pipeline%20is%20used%20to%20assemble,pipeline%20module.&text=GridSearchCV%20is%20used%20to%20optimize,to%20find%20the%20best%20model.
19. Machine Learning Mastery - Automate Machine Learning Workflows with Pipelines in Python and scikit-learn: https://machinelearningmastery.com/automate-machine-learning-workflows-pipelines-python-scikit-learn/
20. O que são Pipelines e para que serve isso (Automatize suas etapas de Machine Learning): https://minerandodados.com.br/o-que-sao-pipelines-e-para-que-serve-isso-automatize-suas-etapas-de-machine-learning/
21. How to do cross-validation when upsampling data: https://kiwidamien.github.io/how-to-do-cross-validation-when-upsampling-data.html
22. Damien Martin's Github Portfolio: https://gist.github.com/kiwidamien
23. imblearn SMOTE documentation: https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html
24. sklearn StandardScaler documentation: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
25. sklearn Cross-Validation documentation: https://scikit-learn.org/stable/modules/cross_validation.html
26. sklearn StratifiedKfold documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
27. sklearn Stratified-Shuffle-Split documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit
28. Why and how to Cross Validate a Model?: https://towardsdatascience.com/why-and-how-to-cross-validate-a-model-d6424b45261f
29. SANJAY Msanjayds' Github Portfoliohttps: //github.com/Msanjayds/Machine_Learning_Projects/blob/master/2.%20CrossValidation.ipynb
30. Machine Learning Mastery - A Gentle Introduction to k-fold Cross-Validation: https://machinelearningmastery.com/k-fold-cross-validation/
31. sklearn Train_Test_Split documentation: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
32. sklearn Metrics documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html, https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
34. Como saber se seu modelo de Machine Learning está funcionando mesmo: https://paulovasconcellos.com.br/como-saber-se-seu-modelo-de-machine-learning-est%C3%A1-funcionando-mesmo-a5892f6468b
35. Cross-Validation in Machine Learning: How to Do It Right: https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right
36. sklearn Logistic Regression documentation: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
37. sklearn Linear Discriminant Analysis documentation: https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
38. sklearn Random Forest Classifier documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

