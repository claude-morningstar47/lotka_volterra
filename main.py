import numpy as np
import matplotlib.pyplot as plt

# Paramètres du modèle
alpha = 2/3
beta = 4/3
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
lapin = np.array(lapin)
renard = np.array(renard)

# Affichage des résultats
plt.figure(figsize=(12, 6))
plt.plot(time, lapin, label='Proies - lapins', color='blue')
plt.plot(time, renard, label='Prédateurs - renards', color='red')
plt.xlabel('Temps')
plt.ylabel('Population')
plt.title('Évolution temporelle des populations')
plt.legend()
plt.grid(True)

plt.show()
