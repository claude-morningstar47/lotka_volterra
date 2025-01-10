import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product


# Fonction pour charger les données CSV
def load_csv_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if {'date', 'lapin', 'renard'}.issubset(data.columns):
            return data['date'].values, data['lapin'].values, data['renard'].values
        else:
            raise ValueError(
                "Le fichier CSV doit contenir les colonnes 'date', 'lapin', et 'renard'.")
    except Exception as e:
        raise IOError(f"Erreur lors du chargement du fichier CSV : {e}")


# Fonction pour simuler la dynamique des populations
def simulate_population_dynamics(alpha, beta, delta, gamma, time, step, n_iterations, lapin_sim, renard_sim):
    for _ in range(n_iterations):
        dlapin_dt = lapin_sim[-1] * \
            (alpha - beta * renard_sim[-1]) * step + lapin_sim[-1]
        drenard_dt = renard_sim[-1] * \
            (delta * lapin_sim[-1] - gamma) * step + renard_sim[-1]

        time.append(time[-1] + step)
        lapin_sim.append(max(0, dlapin_dt))
        renard_sim.append(max(0, drenard_dt))

    return time, lapin_sim, renard_sim


# Fonction pour calculer la MSE
def calculate_mse_for_params(lapin_real, renard_real, lapin_sim, renard_sim, time, step, alpha, beta, delta, gamma, n_iterations):
    time, lapin_sim, renard_sim = simulate_population_dynamics(
        alpha, beta, delta, gamma, time, step, n_iterations, lapin_sim, renard_sim)

    time = np.array(time)
    time = time[:1000]
    lapin_sim = np.array(lapin_sim) * 1000
    renard_sim = np.array(renard_sim) * 1000

    step_ratio = max(1, len(lapin_sim) // len(lapin_real))
    lapin_resized = lapin_sim[::step_ratio][:len(lapin_real)]
    renard_resized = renard_sim[::step_ratio][:len(renard_real)]

    mse_lapin = np.mean((lapin_real - lapin_resized) ** 2)
    mse_renard = np.mean((renard_real - renard_resized) ** 2)

    return mse_lapin, mse_renard, lapin_resized, renard_resized, np.array(time)


# Fonction pour optimiser les paramètres
def optimize_params(lapin_real, renard_real, lapin_sim, renard_sim, time, step, n_iterations, alpha_values, beta_values, delta_values, gamma_values):
    best_params = None
    lowest_mse = float('inf')

    for alpha, beta, delta, gamma in product(alpha_values, beta_values, delta_values, gamma_values):
        mse_lapin, mse_renard, lapin_resized, renard_resized, new_time = calculate_mse_for_params(
            lapin_real, renard_real, lapin_sim, renard_sim, time, step, alpha, beta, delta, gamma, n_iterations)
        mse = mse_lapin + mse_renard
        if mse < lowest_mse:
            lowest_mse = mse
            best_params = (alpha, beta, delta, gamma)
            mse_params = (lowest_mse, mse_lapin, mse_renard)

    return best_params, mse_params, lapin_resized, renard_resized, new_time


# Fonction d'affichage des résultats de simulation
def plot_results(time_real, lapin_real, renard_real, lapin_resized, renard_resized):

    plt.figure(figsize=(15, 6))

    # Graphique des lapins
    plt.plot(time_real, lapin_real, label='Lapins (Réel)',
             color='cyan', linewidth=2)
    plt.plot(time_real, lapin_resized, label='Lapins (Simulé)',
             color='blue', linestyle='--')

    # Graphique des renards
    plt.plot(time_real, renard_real, label='Renards (Réel)',
             color='orange', linewidth=2)
    plt.plot(time_real, renard_resized, label='Renards (Simulé)',
             color='red', linestyle='--')

    plt.xlabel('Temps')
    plt.ylabel('Population')
    plt.title('Comparaison des populations simulées et réelles')
    plt.legend()
    # plt.grid(True)
    plt.show()
