import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

#wczytanie danych
red = pd.read_csv("winequality-red.csv", sep=";")
white = pd.read_csv("winequality-white.csv", sep=";")

#funkcja do trenowania i ewaluacji modelu
def train_and_evaluate(data, wine_type="Wine"):
    #usunięcie kolumny "quality", zostają tylko cechy x
    X = data.drop("quality", axis=1).values

    #pobranie kolumny "quality" jako etykiet y
    y = data["quality"].values

    #ustawienie ziarna losowości
    np.random.seed(42)

    #wylosowanie permutacji indeksów danych
    indices = np.random.permutation(len(X))

    #podział danych 80%/20%
    #obliczenie miejsca podziału
    split = int(0.8 * len(X))

    #indeksy zbioru treningowego
    train_idx = indices[:split]

    #indeksy zbioru walidacyjnego
    val_idx = indices[split:]

    #dane treningowe
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    #naiwny klasyfikator Bayesa (Gaussian)

    class GaussianNaiveBayes:
        def fit(self, X, y):
            #lista unikalnych klas jakości
            self.classes = np.unique(y)

            #słowniki, dla każdej klasy model zapamięta:
            self.mean = {} #średnią cech
            self.var = {} #wariancję cech
            self.prior = {} #jak często ta klasa występuje

            #iteracja po każdej klasie jakości
            for c in self.classes:
                #wybór próbek należących do klasy c
                X_c = X[y == c]

                #średnia każdej cechy dla klasy c
                self.mean[c] = X_c.mean(axis=0)

                #wariancja każdej cechy dla klasy c
                # +1e-9 zabezpiecza przed dzieleniem przez zero
                self.var[c] = X_c.var(axis=0) + 1e-9

                #prawdopodobieństwo klasy c
                self.prior[c] = X_c.shape[0] / X.shape[0]

        def predict(self, X):
            #lista na przewidywane klasy
            predictions = []

            #iteracja po każdej próbce
            for x in X:
                #wynik liczbowy dla każdej klasy, im większy tym bardziej model "lubi" tę klasę
                posteriors = []

                #obliczanie prawdopodobieństwa dla każdej klasy
                for c in self.classes:
                    #"jak często w danych treningowych pojawia się ta jakość"
                    prior_log = np.log(self.prior[c])

                    #logarytm funkcji gęstości Gaussa
                    #"jak bardzo cechy tego wina pasują do średnich cech jakości c"
                    likelihood_log = -0.5 * np.sum(
                        np.log(2 * np.pi * self.var[c]) +
                        ((x - self.mean[c]) ** 2) / self.var[c]
                    )

                    #łączna „ocena” klasy c dla tego wina
                    posteriors.append(prior_log + likelihood_log)

                #wybór klasy z największym prawdopodobieństwem
                predictions.append(self.classes[np.argmax(posteriors)])

            #zwrócenie przewidywań jako tablicy numpy
            return np.array(predictions)

    #trenowanie i predykcja

    #utworzenie obiektu klasyfikatora
    gnb = GaussianNaiveBayes()

    #trenowanie modelu na danych treningowych
    gnb.fit(X_train, y_train)

    #predykcja jakości dla zbioru walidacyjnego
    y_pred = gnb.predict(X_val)

    #obliczenie dokładności (accuracy)
    accuracy = np.mean(y_pred == y_val)

    #macierz pomyłek
    #posortowana lista klas jakości
    classes = np.sort(np.unique(y))

    #obliczenie macierzy pomyłek
    cm = confusion_matrix(y_val, y_pred, labels=classes)

    #wizualizacja macierzy pomyłek
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion matrix – {wine_type}\nAccuracy: {accuracy:.4f}")
    plt.show()

#uruchomienie modelu
train_and_evaluate(red, wine_type="Red Wine")
train_and_evaluate(white, wine_type="White Wine")


def print_priors(data, wine_type="Wine"):
    #liczba wszystkich próbek
    total_samples = len(data)

    #zliczenie klas jakości
    class_counts = data["quality"].value_counts().sort_index()

    print(f"\nPrawdopodobieństwa a priori dla {wine_type}:")
    for quality, count in class_counts.items():
        prior = count / total_samples
        print(f"Jakość {quality}: P(c) = {prior:.4f}")

print_priors(red, "Red Wine")
print_priors(white, "White Wine")
