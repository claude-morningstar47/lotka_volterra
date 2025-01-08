import numpy as np
import pandas as pd
from itertools import product


# Fonction pour charger les données CSV
def load_csv_data(file_path):
    try:
        data = pd.read_csv(file_path)
        if 'date' in data.columns and 'lapin' in data.columns and 'renard' in data.columns:
            return data['date'].values, data['lapin'].values, data['renard'].values
        else:
            raise ValueError(
                "Le fichier CSV doit contenir les colonnes 'date', 'lapin', et 'renard'.")
    except Exception as e:
        raise IOError(f"Erreur lors du chargement du fichier CSV : {e}")


# Fonction pour calculer la MSE
def calculate_mse(real_lapin, real_renard, simulated_lapin, simulated_renard):
    try:
        mse_lapin = np.mean((real_lapin - simulated_lapin) ** 2)
        mse_renard = np.mean((real_renard - simulated_renard) ** 2)
        return mse_lapin, mse_renard
    except Exception as e:
        raise ValueError(f"Erreur dans le calcul du MSE : {e}")


# Fonction pour résoudre le modèle de Lotka-Volterra
def solve_lotka_volterra(f, t_span, y0, params, dt=0.01):

    t = np.arange(t_span[0], t_span[1], dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0

    for i in range(1, len(t)):
        derivatives = f(t[i-1], y[i-1], *params)
        y[i] = y[i-1] + dt * derivatives

    return t, y


# Fonction pour résoudre le modèle de Lotka-Volterra
def optimize_parameters(real_data, t_span, initial_conditions):
    param_values = [1/3, 2/3, 1, 4/3]
    best_mse = float('inf')
    best_params = None

    for alpha, beta, gamma, delta in product(param_values, repeat=4):
        params = (alpha, beta, gamma, delta)
        _, simulated_data = solve_lotka_volterra(
            params, t_span, initial_conditions)

        mse = calculate_mse(real_data, simulated_data)
        if mse < best_mse:
            best_mse = mse
            best_params = params
    return best_params, best_mse
