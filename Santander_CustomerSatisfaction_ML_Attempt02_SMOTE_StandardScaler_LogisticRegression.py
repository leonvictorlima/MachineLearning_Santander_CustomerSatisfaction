### Attempt Number 02 - SMOTE + StandardScaler + LogisticRegression 

# Loading dataset

from pandas import read_csv

df4 = pd.read_csv('df4.csv')

# Libraries

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

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
X_smote, y_smote = smote.fit_resample(X_train, y_train)

## Normalizing by Standard Scaler

## Standarization

standardscaler = StandardScaler()
X_rescaled = standardscaler.fit_transform(X_smote) # Training
X_test_rescaled = standardscaler.fit_transform(X_test) # Test


# Machine Learning

lr = LogisticRegression()

lr.fit(X_rescaled, y_smote)

y_pred_lr = lr.predict(X_test_rescaled)

lr_recall = recall_score(y_test, y_pred_lr, pos_label = 0, average= 'binary')) #TP/TP + FN 
lr_accur = accuracy_score(y_test, y_pred_lr)
lr_cf = confusion_matrix(y_test, y_pred_lr)

prediction = lr.score(X_test_rescaled, y_test)
print("\n\n Logistic Regression score: ",prediction)
print('\n')

print(" Logistic Regression Recall:", lr_recall )
print(" Logistic Regression accuracy:", lr_accur )
print('\n')
print(" Logistic Regression Confunsion Matrix:\n", lr_cf)

### Conclusion 

# This model not resulted in satisfied accuracy.