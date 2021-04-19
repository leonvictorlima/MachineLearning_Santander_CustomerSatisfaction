### Attempt Number 09 - StandardScaler + Cross Validation +Algorithms


# Loading dataset

from pandas import read_csv

df4 = pd.read_csv('df4.csv')

# Libraries

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score ,roc_curve,auc, accuracy_score

# Splitting the data

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

kf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = seed)

# Initialize model

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('RF', RandomForestClassifier()))


# Split data training and test
i=0
for train_index, test_index in kf.split(X_scaled, y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    x_train_fold, x_test_fold = X_scaled[train_index], X_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    i=i+1

# evaluate each model
results = []
names = []

def metrics(true,predicted):
    return roc_auc_score(true,predicted)

def accuracy(true,predicted):
    return accuracy_score(true,predicted)

# Applying models

for name, model in models:
    
        model.fit(x_train_fold,y_train_fold)
        pred=model.predict(x_test_fold)
        res=metrics(y_test_fold,pred)
        acc = accuracy(y_test_fold,pred)
        results.append(res)
        names.append(name)
        msg = "Modelo: %s, ROC-AUC-Score: %f, Accuracy: %f " % (name,res,acc)
        print(msg)

    
### Conclusion

# Trying evaluate the model without PCA and the results were a shame!  
