### Attempt Number 04 - SMOTE + StandardScaler + GaussianNB + Cross- Validation 

# Loading dataset

from pandas import read_csv

df4 = pd.read_csv('df4.csv')

# As using SMOTE, the results is a balanced data. So we can use cross-validation k-fold directly. If our dataset are unbalanced 
# the use of StratifiedKfold is recomended instead. 

from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# Separete the data

X = df4.drop("TARGET", axis=1)
y = df4['TARGET']


# Defining parameters

seed = 42
num_folds = 10
size = 0.30

# SMOTE

## Applying SMOTE - Making dataset balanced

smote = SMOTE(random_state = seed)
X_new, y_new = smote.fit_resample(X, y)

# Printing 
fig, axes = plt.subplots(nrows=2, ncols=1)
pd.Series(y).value_counts().plot(kind='bar',color = ['#1F77B4', '#FF7F0E'], ax=axes[0])
pd.Series(y_new).value_counts().plot.bar(color = ['#1F77B4', '#FF7F0E'], ax=axes[1])

print('\nX Before SMOTE:', X.shape)
print('\n Y Before SMOTE:', y.shape)

print('\n X After SMOTE:',X_new.shape)
print('\n Y After SMOTE:',y_new.shape)
print('\n\n')


# Standarization

standardscaler = StandardScaler() #Instancing object
X_scaled = standardscaler.fit_transform(X_new) # Apply standarization training data

# Split folds

# Used for unbalaced data;
#skf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = seed)

kfold = KFold(num_folds, True, random_state = seed)

# Initialize model

model_NB = GaussianNB()

# Evaluating 

# Cross Validation
#result = cross_val_score(model_NB, X_new, y_new, cv = kfold, scoring = 'accuracy')
result = cross_validate(model_NB, X_new, y_new, cv = kfold, scoring = ['accuracy', 'recall'])
# Results


print("Accuracy: %.3f" % (result['test_accuracy'].mean()*100))
print("Recall: %.3f" % (result['test_recall'].mean()))


### Conclusion 

# This model get diversified results in Accurancy and Recall. In fact, this show that this model is not good one.