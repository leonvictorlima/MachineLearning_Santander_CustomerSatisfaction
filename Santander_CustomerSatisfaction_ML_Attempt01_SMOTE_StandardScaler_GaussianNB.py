
# Loading data

from pandas import read_csv

df4 = pd.read_csv('df4.csv')


## Machine Learning

### Attempt Number 01 - SMOTE + StandardScaler + GaussianNB

# Libraries

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, classification_report


# Defining X and y;

X = df4.iloc[:,0:345]
y = df4.iloc[:,345] #-- TARGET

# Set references

seed = 42
test_size = 0.30

#Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size ,random_state = seed)

## Applying SMOTE - Making dataset balanced

smote = SMOTE(random_state = seed)
X_new, y_new = smote.fit_resample(X_train, y_train)

# Printing 
fig, axes = plt.subplots(nrows=2, ncols=1)
pd.Series(y_train).value_counts().plot(kind='bar',color = ['#1F77B4', '#FF7F0E'], ax=axes[0])
pd.Series(y_new).value_counts().plot.bar(color = ['#1F77B4', '#FF7F0E'], ax=axes[1])

print('\nX Before SMOTE:', X_train.shape)
print('\n Y Before SMOTE:', y_train.shape)

print('\n X After SMOTE:',X_new.shape)
print('\n Y After SMOTE:',y_new.shape)
print('\n\n')


## Normalizing by Standard Scaler

## Standarization

standardscaler = StandardScaler()
X_train_2 = standardscaler.fit_transform(X_new)
X_test_2 = standardscaler.fit_transform(X_test)

print('Original Data\n\n', df4.iloc[0:5,0:5].values)
print('\n\n StandardScaler data\n\n', X_train_2[0:5,0:5])

# Machine Learning

gaussianNB = GaussianNB()

# Model
nb_model = gaussianNB.fit(X_train_2, y_new)

# Prediction
y_pred = nb_model.predict(X_test_2)

# Metrics
nb_recall = recall_score(y_test, y_pred, pos_label = 0, average= 'binary') #TP/TP + FN 
nb_accur = accuracy_score(y_test, y_pred)
cf = confusion_matrix(y_test, y_pred)

# Score Model
prediction = nb_model.score(X_test_2, y_test)

# Classification report
report = classification_report(y_test, y_pred)

print("\n\nNaive Bayes score: ",prediction)
print('\n')

print(" Naive bayes Recall:", nb_recall )
print(" Naive bayes accuracy:", nb_accur )
print('\n')
print(" Confunsion Matrix:\n", cf)
print('\n')
print(report)

### Conclusion 

# This model just learned satisfaction. So, this is not good. It is not  a generalized model. 
# It just made Predictions = yes (or 0, in this case) and did not learn Predictions = No (or 1, in this case). 
# For thar reason, it is biased model.
