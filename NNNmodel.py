import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

df = pd.read_csv("Leads-data.csv")
lost_reason = df['lost_reason']
df = df.drop(['Agent_id','lost_reason','source','source_city','source_country','utm_source','utm_medium','des_city','des_country','lead_id'],axis=1)

df = df.dropna()

df = df[df['status'].isin({'WON','LOST'})]
df = df[df['room_type'].isin({'Ensuite','Entire Place','Studio'})]
df = df[~df['budget'].str.isalpha()]

budget = df['budget']

cleaned_budget=[]
for i in budget:
    for x in i:
        if x.isnumeric():
            cleaned_budget.append(i)
            break

data = cleaned_budget

data_cleaned = [re.sub(r'[^\d.-]', '', value) for value in data]

data_averaged = []
for value in data_cleaned:
    if '-' in value:
        start, end = value.split('-')
        average = (float(start) + float(end)) / 2
        data_averaged.append(average)
    elif value!='':
        data_averaged.append(0)
    else:
        data_averaged.append(value)

cleaned_budget = data_averaged

x=sum(cleaned_budget)/len(cleaned_budget)
for i in range(len(cleaned_budget)):
     if x<=cleaned_budget[i]:
            cleaned_budget[i]=x

cleaned_budget.append(x)
df = df.drop(['budget'],axis=1)
df['new_budget'] = cleaned_budget
df['lost_reason'] = lost_reason

label_encoder = LabelEncoder()
X = df.drop('status', axis=1)
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = label_encoder.fit_transform(X[column])

y = df['status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_matrix)

TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]
TP = confusion_matrix[1, 1]

print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("True Negatives (TN):", TN)
print("False Negatives (FN):", FN)

# Calculate precision
precision = TP / (TP + FP)
print("Precision:", precision)

# Calculate recall
recall = TP / (TP + FN)
print("Recall:", recall)

#calculating F1 score:
F1score=2*((precision*recall)/(precision+recall))

print("F1score is =",F1score)