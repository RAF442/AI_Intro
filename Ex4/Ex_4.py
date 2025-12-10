import autograd.numpy as np
from autograd import grad
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class SolverParameters:
    alpha: float #wspołczynnik kroku gradientu
    max_iter: int = 150 #maksymalna liczba iteracji
    tol_grad: float = 0.000001 #tolerancja dla normy gradientu (stop, jeśli gradient jest bardzo mały)
    tol_step: float = 0.000001 #tolerancja dla zmiany x między iteracjami (stop, jeśli zmiana jest minimalna)

@dataclass
class SolverResult:
    x: np.ndarray            #rozwiązanie
    values: list             #wartości funkcji
    steps: int               #wykonane iteracje

def solver(eval_func, x0, params: SolverParameters):
    gradient = grad(eval_func) #funkcja gradient
    x = np.array(x0, dtype=float) #kopia punktu startowego
    values = [] #przechowuje wartości funkcji w kolejnych iteracjach

    for t in range(params.max_iter): #pętla po iteracjach
        g = gradient(x) #obliczenie gradientu w punkcie x
        values.append(eval_func(x)) #dodanie do listy values wartości funkcji w punkcie x

        if np.linalg.norm(g) < params.tol_grad: #warunek stopu, jeśli norma gradientu jest mniejsza niż tol_grad
            break

        step = -params.alpha * g #obliczenie kroku w kierunku przeciwnym do gradientu

        #zabezpieczenie przed eksplozją normy kroku
        max_step_norm = 100.0
        if np.linalg.norm(step) > max_step_norm:
            step = step / np.linalg.norm(step) * max_step_norm

        x_new = x + step #aktualizacja punktu x o obliczony krok

        x_new = np.clip(x_new, -10, 10) #ograniczenie wartości wektora x_new do przedziału [-10, 10] dla każdego wymiaru

        if np.linalg.norm(x_new - x) < params.tol_step: #warunek stopu, jeśli zmiana x jest mniejsza niż tol_step
            x = x_new
            break

        x = x_new #przypisanie x_new do x i kontynuacja kolejnej iteracji jeśli warunki stopu nie zostały spełnione

    return SolverResult(x=x, values=values, steps=t + 1) #zwrócenie obiektu SolverResult z ostatnim punktem, wartościami funkcji i liczbą wykonanych iteracji

def quadratic(x):
    return 0.5 * np.sum(x**2)
def rosenbrock(x):
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)
def ackley(x):
    n = x.size
    return (
        -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
        - np.exp(np.sum(np.cos(2*np.pi*x)) / n)
        + 20 + np.e
    )

def solve_and_plot(func, name, x0, alphas, max_iter=10): #funkcja rysująca krzywe zbieżności dla różnych wartości alpha
    results = {} #pusty słownik, w którym będą przechowywane przebiegi wartości funkcji

    for a in alphas: #dla każdej wartości alpha generuje punkt startowy x0_run losowo przesunięty o +/-1 wokół x0
        if isinstance(x0, np.ndarray):
            x0_run = x0 + np.random.uniform(-1, 1, size=x0.shape)
        else:
            x0_run = np.array([x0]) + np.random.uniform(-1, 1)

        params = SolverParameters(alpha=a, max_iter=max_iter) #tworzenie obiektu parametrów
        res = solver(func, x0_run, params) #uruchomienie solvera dla danej funkcji i alpha
        results[a] = res.values #zapisanie przebiegu wartości funkcji dla danego alpha

    plt.figure(figsize=(15,10))

    for a in alphas:
        plt.plot(range(len(results[a])), results[a], label=f"α = {a}", linewidth=2)

    plt.title(f"Zbieżność dla funkcji {name}")
    plt.xlabel("t")
    plt.ylabel("q(x_t)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

alphas = [1,10,100]
x0 = np.random.uniform(-10, 10, size=10)

solve_and_plot(quadratic, "Quadratic", x0, alphas)
solve_and_plot(rosenbrock, "Rosenbrock", x0, alphas)
solve_and_plot(ackley, "Ackley", x0, alphas, 300)