### Attempt Number 03 - StandardScaler + GaussianNB + Cross- Validation 

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

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

lst_accur = []

# Initialize model

NB = GaussianNB()

# Evaluating 

for train_index, test_index in skf.split(X, y):
    x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    NB.fit(x_train_fold, y_train_fold)
    lst_accur.append(NB.score(x_test_fold, y_test_fold))
    
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

# Very bad results in this model.