import numpy as np
import matplotlib.pyplot as plt
from lotka_func import load_csv_data, calculate_mse

# Paramètres du modèle
alpha = 1/3
beta = 2/4
delta = 1
gamma = 1

step = 0.001
simulation_time = 100

# Conditions initiales
time = [0]
lapin = [1]
renard = [2]

# Nombre d'itérations nécessaires
n_iterations = int(simulation_time / step)

data_file_path = 'populations_lapins_renards.csv'
time_real, lapin_real, renard_real = load_csv_data(data_file_path)

# Algorithme d'Euler
for _ in range(n_iterations):
    new_value_time = time[-1] + step

    # Équations du modèle de Lotka-Volterra
    dlapin_dt = alpha * lapin[-1] - beta * lapin[-1] * renard[-1]
    drenard_dt = delta * lapin[-1] * renard[-1] - gamma * renard[-1]

    # Mise à jour des populations avec la méthode d'Euler
    new_value_lapin = lapin[-1] + step * dlapin_dt
    new_value_renard = renard[-1] + step * drenard_dt

    # Ajout des nouvelles valeurs aux listes
    time.append(new_value_time)
    lapin.append(max(0, new_value_lapin))
    renard.append(max(0, new_value_renard))

# Conversion en tableaux numpy pour l'efficacité
time = np.array(time)
lapin = np.array(lapin) * 1000
renard = np.array(renard) * 1000

# Redimensionnement des données simulées pour correspondre aux données réelles
if len(lapin) > len(lapin_real):
    step_ratio = len(lapin) // len(lapin_real)
    lapin_resized = lapin[::step_ratio][:len(lapin_real)]
    renard_resized = renard[::step_ratio][:len(renard_real)]
else:
    lapin_resized = lapin[:len(lapin_real)]
    renard_resized = renard[:len(renard_real)]

# Calcul de l'erreur quadratique moyenne
mse_lapin, mse_renard = calculate_mse(
    lapin_real, renard_real, lapin_resized, renard_resized)

print(f"Erreur quadratique moyenne (MSE) pour les lapins : {mse_lapin:.2f}")
print(f"Erreur quadratique moyenne (MSE) pour les renards : {mse_renard:.2f}")

# Affichage des résultats
plt.figure(figsize=(15, 6))
plt.plot(time, lapin, label='Proies - lapins (Simulés)', color='blue')
plt.plot(time, renard, label='Prédateurs - renards (Simulés)', color='red')
plt.scatter(time_real, lapin_real,
            label='Proies - lapins (Réel)', color='cyan', s=10)
plt.scatter(time_real, renard_real,
            label='Prédateurs - renards (Réel)', color='orange', s=10)
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title('Évolution temporelle des populations')
plt.legend()
plt.show()
