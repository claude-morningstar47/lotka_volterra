import numpy as np
import matplotlib.pyplot as plt
from lotka_func import load_csv_data, optimize_params, plot_results

# Charger les données depuis le fichier CSV
data_file_path = 'populations_lapins_renards.csv'
time_real, lapin_real, renard_real = load_csv_data(data_file_path)

# Paramètres du modèle
# alpha_values = [1/3, 2/3, 1, 4/3]
# beta_values = [1/3, 2/3, 1, 4/3]
# delta_values = [1/3, 2/3, 1, 4/3]
# gamma_values = [1/3, 2/3, 1, 4/3]
alpha_values = [0.48111119]
beta_values = [0.87111119]
delta_values = [1.25]
gamma_values = [1.24]

# Paramètres pour l'optimisation
step = 0.00030
simulation_time = len(time_real)
n_iterations = 100_000

# Conditions initiales
time = [0]
lapin_sim = [1]
renard_sim = [2]

# Optimiser les paramètres
best_params, mse_params, lapin_resized, renard_resized, new_time = optimize_params(
    lapin_real, renard_real, lapin_sim, renard_sim, time, step, n_iterations,
    alpha_values, beta_values, delta_values, gamma_values
)

# Extraire les meilleurs paramètres
alpha, beta, delta, gamma = best_params
lowest_mse, mse_lapin, mse_renard = mse_params

# Afficher les résultats de l'optimisation
print(f"Meilleurs paramètres trouvés : alpha={alpha:.2f}, beta={
      beta:.2f}, delta={delta:.2f}, gamma={gamma:.2f}")
print(f'MSE Lapin: {mse_lapin:.2f}, MSE Renard: {mse_renard:.2f}')
print(f"Erreur quadratique moyenne (MSE) : {lowest_mse:.2f}")

# Visualiser les résultats
plot_results(time_real, lapin_real, renard_real, lapin_resized, renard_resized)
