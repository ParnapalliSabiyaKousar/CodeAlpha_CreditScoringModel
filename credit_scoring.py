import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# LOAD DATA
df = pd.read_csv("data/credit_data.csv")

# SHOW DATA
print(df.head())

# CONVERT TEXT COLUMNS TO NUMBERS
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

# USE LAST COLUMN AS TARGET
target = df.columns[-1]

X = df.drop(target, axis=1)
y = df[target]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MODEL
model = RandomForestClassifier()

# TRAIN MODEL
model.fit(X_train, y_train)

# PREDICT
y_pred = model.predict(X_test)

# RESULTS
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Confusion Matrix")

plt.show()

import pickle

# SAVE MODEL
with open("model/credit_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")