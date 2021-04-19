### Attempt Number 07 - SMOTE + StandardScaler + Logistic Regression + Cross- Validation - Another way

# Loading dataset

from pandas import read_csv

df4 = pd.read_csv('df4.csv')

# Libraries

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

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

# Cross_validation

skf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = seed)

# Model

lr = LogisticRegression(C=2, random_state=seed)

lst_score = []
i=1
# Evaluating 

for train_index, test_index in skf.split(X, y):
    print('{} of KFold {}'.format(i,skf.n_splits))
    x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    
    lr.fit(x_train_fold, y_train_fold)
    score = roc_auc_score(y_test_fold,lr.predict(x_test_fold))
    print('ROC AUC score:',score)
    lst_score.append(score)   
    i+=1

    
print('Confusion matrix\n',confusion_matrix(y_test_fold,lr.predict(x_test_fold)))
print('Cv',lst_score,'\nMean cv Score',np.mean(lst_score))    
print('Accuracy', lr.score(x_test_fold,y_test_fold))

### Conclusion 

# The model's result is the same up now! 