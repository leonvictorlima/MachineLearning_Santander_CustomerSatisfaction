### Attempt Number 05 - StandardScaler + Logistic Regression + Cross- Validation 

# Loading dataset

from pandas import read_csv

df4 = pd.read_csv('df4.csv')

# Libraries

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc

# Separete the data

X = df4.drop("TARGET", axis=1)
y = df4['TARGET']


# Defining parameters

seed = 42
num_folds = 5
size = 0.30

# Standarization

standardscaler = StandardScaler() #Instancing object
X_scaled = standardscaler.fit_transform(X) # Apply standarization training data

# Cross_validation

skf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = seed)


# Initialize model

lr = LogisticRegression(C=2)

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

    
print('\nConfusion matrix\n',confusion_matrix(y_test_fold,lr.predict(x_test_fold)))
print('\nCv',lst_score,'\nMean cv Score',np.mean(lst_score))
print('\nAccuracy: ', lr.score(x_test_fold,y_test_fold))


### Conclusion 

# This model got diversified results in Accurancy and Recall and this is not good.