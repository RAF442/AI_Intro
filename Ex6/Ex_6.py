import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt


def boltzmann_action(Q_row, temperature):

    #wybór akcji zgodnie z rozkładem Boltzmanna
    Q_scaled = Q_row / temperature #skalowanie wartości Q temperaturą
    Q_scaled -= np.max(Q_scaled)  #stabilizacja numeryczna

    probs = np.exp(Q_scaled) #obliczenie funkcji wykładniczej dla każdej akcji
    probs /= np.sum(probs) #normalizacja – suma prawdopodobieństw = 1

    return np.random.choice(len(Q_row), p=probs) #Losowy wybór akcji zgodnie z obliczonym rozkładem


def q_learning(
    env, #środowisko
    episodes=1250, #liczba epizodów uczenia
    alpha=0.5, #współczynnik uczenia
    gamma=0.99, #współczynnik dyskontowania
    epsilon=1.0, #początkowe ε (eksploracja)
    epsilon_min=0.01, #minimalna wartość ε
    epsilon_decay=0.999, #tempo zmniejszania ε
    strategy="epsilon_greedy", #strategia eksploracji
    temperature=1.0 #temperatura (Boltzmann)
):
    #inicjalizacja tabeli Q zerami
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    #lista do zapisu nagród z każdego epizodu
    rewards_per_episode = []

    #pętla po epizodach
    for episode in range(episodes):
        #reset środowiska – pobranie stanu początkowego
        state, _ = env.reset()
        #flaga zakończenia epizodu
        done = False
        #suma nagród w epizodzie
        total_reward = 0

        #pętla kroków w epizodzie
        while not done:

            #wybór akcji
            if strategy == "epsilon_greedy":
                #losowa akcja z prawdopodobieństwem ε
                if random.random() < epsilon:
                    action = env.action_space.sample()
                #najlepsza znana akcja z tablicy Q
                else:
                    action = np.argmax(Q[state])

            elif strategy == "boltzmann":
                #wybór akcji metodą softmax
                action = boltzmann_action(Q[state], temperature)

            #wykonanie akcji w środowisku
            next_state, reward, terminated, truncated, _ = env.step(action)
            #sprawdzenie czy epizod się zakończył
            done = terminated or truncated

            #aktualizacja
            #wzór Q-learningu
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            #przejście do następnego stanu
            state = next_state
            #dodanie nagrody do sumy
            total_reward += reward

        #zmniejszanie ε po każdym epizodzie
        if strategy == "epsilon_greedy":
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

        #zapis nagrody z epizodu
        rewards_per_episode.append(total_reward)

    #zwrócenie wyuczonej tablicy Q i historii nagród
    return Q, rewards_per_episode

#alpha
def experiment_alpha():
    #utworzenie środowiska Taxi
    env = gym.make("Taxi-v3")
    #testowane wartości współczynnika uczenia
    alphas = [0.1, 0.5, 0.9]

    plt.figure(figsize=(15, 10))

    #Pętla po wartościach alpha
    for alpha in alphas:
        #trening agenta
        _, rewards = q_learning(
            env,
            alpha=alpha,
            strategy="epsilon_greedy",
            epsilon=1.0,
            epsilon_decay=0.999
        )

        #średnia krocząca z 100 epizodów
        avg = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        plt.plot(avg, label=f"α = {alpha}")

    plt.title("Wpływ współczynnika uczenia α")
    plt.xlabel("Epizody")
    plt.ylabel("Średnia nagroda (100 ep.)")
    plt.legend()
    plt.grid()
    plt.show()


#strategie
def experiment_strategy():
    #utworzenie środowiska Taxi
    env = gym.make("Taxi-v3")

    #konfiguracje eksperymentu
    configs = [
        ("Stały ε = 0.1",
         dict(strategy="epsilon_greedy", epsilon=0.1, epsilon_decay=1.0)),

        ("Wolno malejący ε",
         dict(strategy="epsilon_greedy", epsilon=1.0, epsilon_decay=0.999)),

        ("Szybko malejący ε",
         dict(strategy="epsilon_greedy", epsilon=1.0, epsilon_decay=0.99)),

        ("Boltzmann (T = 1.0)",
         dict(strategy="boltzmann", temperature=1.0)),
    ]

    plt.figure(figsize=(15, 10))

    #test każdej strategii
    for name, params in configs:
        _, rewards = q_learning(env, **params)
        #średnia krocząca
        avg = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        plt.plot(avg, label=name)

    plt.title("Porównanie strategii")
    plt.xlabel("Epizody")
    plt.ylabel("Średnia nagroda (100 ep.)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    experiment_alpha()
    experiment_strategy()