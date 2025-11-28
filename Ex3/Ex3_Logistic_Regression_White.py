import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

data = pd.read_csv('winequality-white.csv', sep=';')

data['quality_binary'] = (data['quality'] >= 7).astype(int)

X = data.drop(['quality', 'quality_binary'], axis=1)
y = data['quality_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(C=1, max_iter=100)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

#macierz konfuzji
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"White wine confusion matrix for logistic regression (accuracy = {acc:.2f})")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.show()