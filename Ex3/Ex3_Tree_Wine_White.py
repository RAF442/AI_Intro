import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

data = pd.read_csv('winequality-white.csv', sep=';') #ładowanie danych
data['quality_binary'] = (data['quality'] >= 7).astype(int) #wino dobre, ocena powyżej 7

X = data.drop(['quality', 'quality_binary'], axis=1) #cechy
y = data['quality_binary'] #etykiety

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #zbiory treningowy i testowy

#klasa węzła drzewa
class DecisionTreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth #głębokość
        self.max_depth = max_depth #maksymalna głębokość drzewa
        self.feature_index = None #numer kolumny cechy, po której dokonujemy podziału
        self.threshold = None #wartość progu podziału
        self.left = None #wskaźnik do węzła potomnego lewego
        self.right = None #wskażnik do wezła potomnego prawego
        self.label = None #jeśli węzeł jest liściem, etykieta klasy (0 lub 1)

#funkcja entropii
def entropy(y):
    if len(y) == 0:
        return 0 #pusta lista, entropia 0
    p1 = np.mean(y) #prawdopodobieństwo klasy 1
    if p1 == 0 or p1 == 1:
        return 0 #jeśli wszystkie etykiety są takie same, entropia 0
    return -p1*np.log2(p1) - (1-p1)*np.log2(1-p1) #entropia binarna

#zysk informacyjny
def information_gain(y, y_left, y_right):
    return entropy(y) - (len(y_left)/len(y))*entropy(y_left) - (len(y_right)/len(y))*entropy(y_right) #definicja ig

#główna klasa drzewa decyzyjnego
class DecisionTreeClassifierCustom:
    def __init__(self, max_depth, min_samples_split):
        self.max_depth = max_depth #maksymalna głębokość drzewa
        self.min_samples_split = min_samples_split #minimalna liczba próbek w węźle, aby dzielić dalej
        self.root = None #korzeń drzewa

    def fit(self, X, y): #trenuje model
        self.root = self._build_tree(X, y) #tworzy drzewo rekurencyjnie, zaczynając od korzenia

    def _build_tree(self, X, y, depth=0): #rekurencyjna funkcja budująca drzewo
        node = DecisionTreeNode(depth=depth, max_depth=self.max_depth) #tworzony nowy węzeł na biężącym poziomie

        if depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1: #zakończenie dzielenia gdy: przekroczono maksymalną głebokość, za mało próbek, wszystkie etykiety są takie same
            node.label = np.round(np.mean(y)) #etykieta liścia = większość głosów
            return node

        best_gain = -1 #inicjalizacja szukania najlepszego podzialu
        best_feature, best_threshold = None, None

        for feature_index in range(X.shape[1]): #iteracja przez wszytkie kolumny cech
            thresholds = np.unique(X[:, feature_index]) #każda unikalna wartość może być prgiem podziału
            for t in thresholds: #sprawdzenie każdego progu
                left_mask = X[:, feature_index] <= t #lewa maska
                right_mask = X[:, feature_index] > t #prawa maska
                y_left, y_right = y[left_mask], y[right_mask] #etykiety
                if len(y_left) == 0 or len(y_right) == 0: #pominięcie jeśli którakolwiek strona jest pusta
                    continue
                gain = information_gain(y, y_left, y_right) #obliczenie ig
                if gain > best_gain: #zapamiętanie jeśli lepsze niż dotychczas
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = t

        if best_gain == -1: #brak znalezienia dobrego podziału
            node.label = np.round(np.mean(y))
            return node

        #zapisanie najlepszego podziału
        node.feature_index = best_feature
        node.threshold = best_threshold

        #dzielenie danych i rekurencyjne budowanie poddrzew
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth+1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth+1)
        return node

    def predict_single(self, x): #przewidywanie dla pojedynczej próbki
        node = self.root
        while node.label is None:
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.label

    def predict(self, X): #przewidywanie dla wielu próbek naraz
        return np.array([self.predict_single(x) for x in X])

max_depth_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
min_samples_splits = [2, 5, 10, 50, 100, 300, 500, 1000]

plt.figure(figsize=(10, 6))

for mss in min_samples_splits:
    accuracies = []

    for depth in max_depth_values:
        tree = DecisionTreeClassifierCustom(max_depth=depth, min_samples_split=mss)
        tree.fit(X_train.values, y_train.values)

        y_pred = tree.predict(X_test.values)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

    plt.plot(max_depth_values, accuracies, marker='o', linestyle='-', label=f"min_samples_split={mss}")

#wykres
plt.title("White wine decision tree accuracy for different values of max depth and minimum samples split")
plt.xlabel("Max depth")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()

#wybranie wartości do macierzy konfuzji
selected_max_depth = 18
selected_min_samples_split = 10

#tworzenie i trenowanie drzewa z wybranymi parametrami
tree = DecisionTreeClassifierCustom(
    max_depth=selected_max_depth,
    min_samples_split=selected_min_samples_split
)

tree.fit(X_train.values, y_train.values)

#predykcja
y_pred_selected = tree.predict(X_test.values)

#acuracy info
acc_selected = accuracy_score(y_test, y_pred_selected)

#macierz konfuzji
cm = confusion_matrix(y_test, y_pred_selected)

plt.figure(figsize=(10, 9))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion matrix for Decision Tree (max_depth={selected_max_depth}, min_samples_split={selected_min_samples_split})")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.show()