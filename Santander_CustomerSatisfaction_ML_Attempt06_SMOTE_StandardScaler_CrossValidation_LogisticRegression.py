### Attempt Number 06 - SMOTE + StandardScaler + Logistic Regression + Cross- Validation 

# Loading dataset

from pandas import read_csv

df4 = pd.read_csv('df4.csv')

# libraries

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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

lst_accur = []

# Initialize model

lr = LogisticRegression()

# Evaluating 

for train_index, test_index in skf.split(X, y):
    x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    lr.fit(x_train_fold, y_train_fold)
    lst_accur.append(lr.score(x_test_fold, y_test_fold))
    
# Results

print('List of possible accuracy:', lst_accur)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(lst_accur)*100, '%')
print('\nMinimum Accuracy:',
      min(lst_accur)*100, '%')
print('\nOverall Accuracy:',
      np.mean(lst_accur)*100, '%')
print('\nStandard Deviation is:', np.std(lst_accur))

### Conclusion 

# The first model with high accuracy? It is wrong...! 