# The chosen one!

### Attempt Number 08 - SMOTE + MaxMinScaler + PCA + Some Algorithms

# Technique for use with all dataset. So, for this dataset we will not remove the list of columns selected before by correlation of Pearson. # Just var_15 column will be removed cause HotEncoderOne technique was used for its value and PCA will be applied;


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


# Loading libraries 

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report


# Splitting data

X = df5.drop('TARGET', axis=1)
y = df5.TARGET #-- TARGET

#Balancing the data - SMOTE

smote = SMOTE(random_state = 42)
X_new, y_new = smote.fit_resample(X, y)

# Plots

fig, axes = plt.subplots(nrows=2, ncols=1)
pd.Series(y).value_counts().plot(kind = 'bar', color = ['#1F77B4', '#FF7F0E'], ax=axes[0])
pd.Series(y_new).value_counts().plot(kind = 'bar', color = ['#1F77B4', '#FF7F0E'], ax=axes[1])

print('\nX Before SMOTE:', X.shape)
print('\n Y Before SMOTE:', y.shape)

print('\n X After SMOTE:',X_new.shape)
print('\n Y After SMOTE:',y_new.shape)
print('\n\n')

# Normalizing data

scaler = MinMaxScaler(feature_range = (0, 1))
X_rescaled = scaler.fit_transform(X_new)

## Cross-Validation

# Split data (Kfold because it is not more a unbalaced dataset)

kf = KFold(n_splits=5,random_state=1,shuffle=True)

for train_index, test_index in kf.split(X_rescaled, y_new):
    x_train_fold, x_test_fold = X_rescaled[train_index], X_rescaled[test_index]
    y_train_fold, y_test_fold = y_new[train_index], y_new[test_index]

# PCA

pca = PCA(n_components = 100)
X_train_pca = pca.fit_transform(x_train_fold)
X_test_pca = pca.fit_transform(x_test_fold)

print('\n',X_train_pca.shape)
print('\n',X_test_pca.shape)

# Defining and Creating a list to storage of some Machine Learning models

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

# Applying models

for name, model in models:
    
        model.fit(X_train_pca,y_train_fold)
        pred=model.predict(X_test_pca)
        res=metrics(y_test_fold,pred)
        acc=accuracy(y_test_fold,pred)
        results.append(res)
        names.append(name)
        msg = "Model %s, ROC_AUC = %f, Accuracy = %f " % (name,res,acc)
        print(msg)


### Conclusion

# This is the model chosen. The results converge in AUC and Accuracy; 

# The Logistic Regression showed the best perfomance compared others techniques applied!! 