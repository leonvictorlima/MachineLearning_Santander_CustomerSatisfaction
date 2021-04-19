### Attempt Number 10 - MaxMinScaler + PCA + Some Algorithms

# Loading dataset

df3 = pd.read_csv('df3.csv')
df3.head(5)

# Taking out some columns to apply PCA

t_columns = ['var15',"var15_range","Labels_var15_range"]


# Function to delete columns from dataframe

def dataframe_column_delete(dataframe, columns_list):
    dataframe_copy = dataframe.copy() 
    for col in columns_list:
        if col in dataframe_copy:
            del dataframe_copy[col]
    return dataframe_copy


# Calling function and creating new dataframe

df5 = dataframe_column_delete(df3,t_columns)
df5 = df5.drop('ID', axis=1)

# Libraries

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# Split data - Should be df3 without 3 or 4 columns
X = df5.drop('TARGET', axis=1)
y = df5.TARGET #-- TARGET

# Seed

seed = 42
num_fols = 10

# Get columns name and index values

columns=X.columns
ind=X.index

# Normalizing data

scaler = MinMaxScaler(feature_range = (0, 1))
X_scaler = scaler.fit_transform(X)

# Transforming to DataFrame

X_scaler = pd.DataFrame(X_scaler, columns = columns, index = ind)
#X_scaler.head(5)

# Cross Validation - KStratified

kf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = seed)

for train_index, test_index in kf.split(X_scaler, y):
    x_train_fold, x_test_fold = X_scaler.iloc[train_index], X_scaler.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

# PCA

pca = PCA(n_components = 57)
X_pca_train = pca.fit_transform(x_train_fold)
X_pca_test = pca.fit_transform(x_test_fold)

print('\n',X_pca_train.shape)
print('\n',X_pca_test.shape)

# Set models and create an empty list to store it;

models = []
models.append(('Logistic Regression', LogisticRegression()))
models.append(('Naive Bayes', GaussianNB()))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('RF', RandomForestClassifier()))

# evaluate each model
results = []
names = []

def metrics(true,predicted):
    return roc_auc_score(true,predicted)

def accuracy(true,predicted):
    return accuracy_score(true,predicted)

# Initialize models;

for name, model in models:
    
        model.fit(X_pca_train,y_train_fold)
        pred=model.predict(X_pca_test)
        res=metrics(y_test_fold,pred)
        acc=accuracy_score(y_test_fold,pred)
        results.append(res)
        names.append(name)
        msg = "Model: %s, ROC_AUC_Score = %f, Accuracy = %f" % (name,res,acc)
        print(msg)


### Conclusion

# That possibilitie did not have the outcomes desired! 