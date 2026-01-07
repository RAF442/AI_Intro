import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

#aproksymowana funkcja
def f(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)

#ziarno losowości
np.random.seed(42)

#liczba próbek uczących i testowych
N_TRAIN = 200
N_TEST = 400

#losowe dane uczące z przedziału [-10, 10]
X_train = np.random.uniform(-10, 10, (N_TRAIN, 1))
y_train = f(X_train)

#dane testowe - równomierne rozmieszczenie na osi X
X_test = np.linspace(-10, 10, N_TEST).reshape(-1, 1)
y_test = f(X_test)

#implementacja sieci neuronowej
#sieć posiada jedną warstwę ukrytą z funkcją aktywacji tanh
class MLP:
    def __init__(self, hidden):
        #liczba neuronów w warstwie ukrytej
        self.h = hidden

        #wagi i bias warstwy wejściowej
        self.W1 = np.random.randn(hidden, 1) * 0.1
        self.b1 = np.zeros((hidden, 1))

        #wagi i bias warstwy ukrytej
        self.W2 = np.random.randn(1, hidden) * 0.1
        self.b2 = np.zeros((1, 1))

    #przejście w przód (forward pass)
    def forward(self, X):
        #obliczenie sygnalów w warstwie ukrytej
        Z1 = self.W1 @ X.T + self.b1
        A1 = np.tanh(Z1)

        #obliczenie wyjścia sieci (bez funkcji aktywacji)
        Z2 = self.W2 @ A1 + self.b2

        #zwracane są wartości pośrednie potrzebne do obliczeń gradientu
        return Z1, A1, Z2.T

    #predykcja wyjścia dla danych wejściowych
    def predict(self, X):
        return self.forward(X)[2]

    #zamiana wszystkich parametrów sieci na jeden wektor (potrzebne do metody ewolucyjnej)
    def params_to_vector(self):
        return np.concatenate([
            self.W1.ravel(), self.b1.ravel(),
            self.W2.ravel(), self.b2.ravel()
        ])

    #odtworzenie parametrów sieci z wektora
    def vector_to_params(self, v):
        h = self.h
        i = 0
        self.W1 = v[i:i+h].reshape(h,1); i += h
        self.b1 = v[i:i+h].reshape(h,1); i += h
        self.W2 = v[i:i+h].reshape(1,h); i += h
        self.b2 = v[i:].reshape(1,1)

#funkcja blędu, średni bląd kwadratowy (Mean Squared Error)
def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

# gradient descent, ręczna implementacja algorytmu spadku gradientu (backpropagation)
def train_gradient(model, X, y, epochs=3000, lr=0.01):
    N = X.shape[0]

    for _ in range(epochs):
        #przejście w przód
        Z1, A1, out = model.forward(X)

        #pochodna funkcji kosztu względem wyjścia
        dY = 2 * (out - y) / N

        #gradienty warstwy wyjściowej
        dW2 = dY.T @ A1.T
        db2 = np.sum(dY.T, axis=1, keepdims=True)

        #propagacja wsteczna do warstwy ukrytej
        dA1 = model.W2.T @ dY.T
        dZ1 = dA1 * (1 - np.tanh(Z1)**2)

        #gradienty warstwy wejściowej
        dW1 = dZ1 @ X
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        #aktualizacja wag i biasów
        model.W1 -= lr * dW1
        model.b1 -= lr * db1
        model.W2 -= lr * dW2
        model.b2 -= lr * db2

    return model

#metoda ewolucyjna (random search), optymalizacja parametrów sieci metodą losowego przeszukiwania
def evolutionary_search(base_model, X, y, candidates=30000, sigma=3):
    #bazowy wektor parametrów
    base_vec = base_model.params_to_vector()

    best_vec = base_vec
    best_loss = np.inf

    #losowe generowanie kandydatów
    for _ in range(candidates):
        v = base_vec + np.random.randn(len(base_vec)) * sigma

        #utworzenie nowego modelu z losowymi parametrami
        m = deepcopy(base_model)
        m.vector_to_params(v)

        #obliczenie blędu
        loss = mse(y, m.predict(X))

        #zachowanie najlepszego rozwiązania
        if loss < best_loss:
            best_loss = loss
            best_vec = v

    #zwrócenie najlepszego znalezionego modelu
    best_model = deepcopy(base_model)
    best_model.vector_to_params(best_vec)
    return best_model

#różne liczby neuronów w warstwie ukrytej
hidden_sizes = [5, 10, 20, 30, 40, 50]
results = []

for h in hidden_sizes:
    base = MLP(h)

    #uczenie metodą gradientową
    grad_model = train_gradient(deepcopy(base), X_train, y_train)
    grad_pred = grad_model.predict(X_test)
    grad_mse = mse(y_test, grad_pred)

    #uczenie metodą ewolucyjną
    evo_model = evolutionary_search(deepcopy(base), X_train, y_train)
    evo_pred = evo_model.predict(X_test)
    evo_mse = mse(y_test, evo_pred)

    #zapis wyników
    results.append((h, grad_mse, evo_mse, grad_pred, evo_pred))

#wykresy aproksymacji, porównanie jakości aproksymacji na danych testowych
for h, gm, em, gp, ep in results:
    plt.figure(figsize=(8,4))
    plt.plot(X_test, y_test, label="f(x)", linewidth=2)
    plt.plot(X_test, gp, label=f"gradient (MSE={gm:.1f})")
    plt.plot(X_test, ep, label=f"ewolucyjna (MSE={em:.1f})")
    plt.title(f"Liczba neuronów w warstwie ukrytej: {h}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()