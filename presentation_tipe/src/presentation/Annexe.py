import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium
from scipy.optimize import minimize
from tick.hawkes import SimuHawkesExpKernels
from collections import Counter


def simulate_hawkes_process_ogata(lmbda0, alpha, beta, T):
    # Initialisation du processus
    t = 0
    event_times = []

    while True:
        # Calcul de la borne superieure de l'intensite
        lmbda_bar = lmbda0 + alpha * len(event_times)

        # Generation d'un temps d'attente
        w = -np.log(np.random.uniform()) / lmbda_bar
        t = t + w

        if t > T:
            break

        # Calcul de l'intensite a l'instant t
        lmbda_t = lmbda0 + alpha * sum(np.exp(-beta * (t - ti)) for ti in event_times)

        # Generation d'une valeur aleatoire uniforme
        u = np.random.uniform()

        # Si u est inferieur ou egal au ratio de l'intensite sur sa borne superieure,
        # on accepte le temps d'attente comme le prochain temps d'evenement du processus de Hawkes
        if u <= lmbda_t / lmbda_bar:
            event_times.append(t)

    # Etape 2 : Calcul de l'intensite a chaque instant
    t_values = np.linspace(0, T, 5000)
    intensity_values = []
    for t in t_values:
        lmbda = lmbda0 + alpha * sum(np.exp(-beta * (t - ti)) for ti in event_times if ti < t)
        intensity_values.append(lmbda)
        
    return t_values, intensity_values, event_times


def filter_burglary(filename):
    # lire le fichier csv
    df = pd.read_csv(filename, delimiter=';', error_bad_lines=False)
    
    # filtrer le dataframe pour ne garder que les lignes ou le 'Primary Type' est 'BURGLARY'
    df = df[df['Primary Type'] == 'BURGLARY']

    # ne garder que les colonnes Date, Primary Type, Latitude, Longitude
    df = df[['Date', 'Primary Type', 'Latitude', 'Longitude']]

    # supprimer les lignes avec des valeurs manquantes
    df = df.dropna()
    
    # Convertir la colonne "Date" en format datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %I:%M:%S %p')

    # Modifier le format de la colonne "Date" pour afficher les dates avec l'heure en format 24h
    df['Date'] = df['Date'].dt.strftime('%m/%d/%Y %H:%M:%S')
    
    # Trier le dataframe par la colonne "Date"
    df = df.sort_values(by='Date')

    # Reinitialiser les index du dataframe trie
    df = df.reset_index(drop=True)

    # enregistrer le nouveau dataframe dans un nouveau fichier csv
    df.to_csv('Crimes2020_trie.csv', index=False)
    

def filter_and_display_burglaries(filename, map_bounds):
    # lire le fichier csv
    df = pd.read_csv(filename, delimiter=',', error_bad_lines=False)

    # filtrer pour ne garder que les cambriolages dans la zone
    df = df[(df['Latitude'] >= map_bounds[0][0]) & (df['Latitude'] <= map_bounds[1][0]) &
             (df['Longitude'] >= map_bounds[0][1]) & (df['Longitude'] <= map_bounds[1][1])]

    # enregistrer le nouveau dataframe dans un nouveau fichier csv
    df.to_csv('Crimes2020_resized.csv', index=False)
    

def get_burglary_times(filename, start_time='01/01/2020 00:00:00'):

    # Charger le nouveau fichier CSV en tant que DataFrame
    df = pd.read_csv(filename)

    # Convertir la colonne de date en datetime
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y %H:%M:%S')
    
    # Definir le temps de reference
    ref_time = pd.to_datetime(start_time, format='%m/%d/%Y %H:%M:%S')

    # Convertir les dates de cambriolage en secondes par rapport au temps de reference
    burglary_times = round((df['Date'] - ref_time).dt.total_seconds() / 86400)

    return burglary_times.values.astype(int)


def display_burglaries_on_map(filename, map_bounds):
    # lire le fichier csv
    df = pd.read_csv(filename, delimiter=',', error_bad_lines=False)

    # filtrer le dataframe pour ne garder que les lignes ou le 'Primary Type' est 'BURGLARY'
    df = df[df['Primary Type'] == 'BURGLARY']

    # ne garder que les colonnes Date, Primary Type, Latitude, Longitude
    df = df[['Date', 'Primary Type', 'Latitude', 'Longitude']]

    # supprimer les lignes avec des valeurs manquantes
    df = df.dropna()

    # definir le centre de la carte comme la moyenne des latitudes et longitudes
    map_center = [df['Latitude'].mean(), df['Longitude'].mean()]

    # creer la carte avec folium
    map_burglaries = folium.Map(location=map_center, zoom_start=13)

    # ajouter des marqueurs pour chaque cambriolage
    for _, row in df.iterrows():
        folium.Marker(location=[row['Latitude'], row['Longitude']], popup=row['Date']).add_to(map_burglaries)

    # definir les limites de la carte
    map_burglaries.fit_bounds(map_bounds)

    # afficher la carte
    return map_burglaries


# Definition d'une fonction recursive pour le calcul de r, un terme utilise dans la fonction de vraisemblance
def _recursive(timestamps, beta):
    r_array = np.zeros(len(timestamps))
    for i in range(1, len(timestamps)):
        r_array[i] = np.exp(-beta * (timestamps[i] - timestamps[i - 1])) * (1 + r_array[i - 1])
    return r_array

# Definition de la fonction de log-vraisemblance specifiant les differents parametres :
def log_likelihood(timestamps, mu, alpha, beta, runtime):
    r = _recursive(timestamps, beta)
    return -runtime * mu + alpha * np.sum(np.exp(-beta * (runtime - timestamps)) - 1) + \
           np.sum(np.log(mu + alpha * beta * r))

# Simulation de donnees Hawkes en utilisant la bibliotheque tick :
mu = 1.2
alpha = 0.6
beta = 0.8
rt = 365

simu = SimuHawkesExpKernels([[alpha]], beta, [mu], rt)
simu.simulate()
t = simu.timestamps[0]

# Definition d'une nouvelle fonction a utiliser par la fonction minimize et qui renvoie la log-vraisemblance negative :
def crit(params, *args):
    mu, alpha, beta = params
    timestamps, runtime = args
    return -log_likelihood(timestamps, mu, alpha, beta, runtime)

# Minimisation de la fonction crit :
minimize(crit, [0.5, 0.5, 0.5], args=(burglary_times, rt), bounds = ((1e-10, None), (1e-10, None), (1e-10, None)), method = 'Nelder-Mead')


def estimate_hawkes_parameters(temps_survenance, duree):
    result = minimize(crit, [0.1, 0.1, 0.1], args=(temps_survenance, duree), bounds = ((1e-10, None), (1e-10, None), (1e-10, None)), method = 'Nelder-Mead')
    return result.x


def plot_burglaries(real_times, simulated_times):
    # Convertir les jours en mois
    real_times_month = np.floor(real_times / 30).astype(int)
    simulated_times_month = np.floor(simulated_times / 30).astype(int)

    # Compter le nombre de cambriolages par mois
    real_counts = Counter(real_times_month)
    simulated_counts = Counter(simulated_times_month)

    # Trouver les mois pour lesquels nous avons des donnees
    all_months = sorted(set(real_counts.keys()).union(set(simulated_counts.keys())))

    # Creer les positions des barres sur l'axe des x
    bar_width = 0.35
    real_positions = np.arange(len(all_months))
    simulated_positions = [x + bar_width for x in real_positions]

    # Creer les histogrammes
    plt.bar(real_positions, [real_counts[month] for month in all_months], width=bar_width, alpha=0.7, label='Reel', color='blue')
    plt.bar(simulated_positions, [simulated_counts[month] for month in all_months], width=bar_width, alpha=0.7, label='Simule', color='red')

    # Ajouter une legende
    plt.legend()

    # Etiquettes des axes
    plt.xlabel('Mois')
    plt.ylabel('Nombre de cambriolages')

    # Ajouter les etiquettes des mois sur l'axe des x
    plt.xticks([r + bar_width / 2 for r in range(len(all_months))], all_months)

    # Titre du graphe
    plt.title('Nombre de cambriolages par mois')

    # Afficher le graphe
    plt.show()

plot_burglaries(burglary_times, timestamps)


def Calibrage(temps, p, n_simulations=100):
    # Calculer le nombre de periodes
    k = int(np.max(temps) // p)

    R = np.zeros(k)     # Nombre reel d'evenements
    S = np.zeros((k, n_simulations))     # Nombre theorique d'evenements
    Diff = np.zeros(k)  # Pourcentage d'erreur

    # Partitionner les donnees en differentes periodes et calibrer un processus de Hawkes pour chaque periode
    for i in range(k):
        # Obtenir les temps d'evenements pour cette periode
        temps_periode = temps[(temps >= i*p) & (temps < (i+1)*p)] - i*p

        # Compter le nombre reel d'evenements
        R[i] = len(temps_periode)

        # Estimer les parametres du processus de Hawkes
        params = estimate_hawkes_parameters(temps_periode, p)

        # Simuler le processus de Hawkes plusieurs fois avec les parametres estimes et compter le nombre d'evenements
        for j in range(n_simulations):
            timestamps = simulate_hawkes_process(params[0], params[1], params[2], p)
            S[i, j] = len(timestamps)

        # Calculer le pourcentage d'erreur
        Diff[i] = np.abs(np.mean(S[i]) - R[i]) / R[i] * 100

    # Calculer l'esperance et l'ecart-type du nombre de cambriolages simules
    S_mean = np.mean(S, axis=1)
    S_std = np.std(S, axis=1)

    # Calculer l'intervalle de confiance a 95% pour l'esperance du nombre de cambriolages simules
    z = 1.96  # z-score pour un intervalle de confiance a 95%
    CI_lower = S_mean - z * S_std / np.sqrt(n_simulations)
    CI_upper = S_mean + z * S_std / np.sqrt(n_simulations)

    return R, S_mean, Diff, (CI_lower, CI_upper)


def optimiser_periodes(temps, p_max=365):
    # Initialiser l'erreur minimale et le nombre optimal de periodes
    erreur_min = float('inf')
    p_optimal = 1

    # Parcourir chaque nombre possible de periodes
    for p in range(1, p_max+1):
        # Calibrer le processus de Hawkes pour le nombre actuel de periodes
        R, S, Diff, (CI_lower, CI_upper) = Calibrage(temps, p)

        # Calculer l'erreur moyenne
        erreur_moyenne = np.mean(Diff)

        # Si l'erreur moyenne est inferieure a l'erreur minimale actuelle, mettre a jour l'erreur minimale et le nombre optimal de periodes
        if erreur_moyenne < erreur_min:
            erreur_min = erreur_moyenne
            p_optimal = p

    return p_optimal, erreur_min


def predict_hawkes_process(mu, alpha, beta, start_time, end_time):
    # Creation de l'objet de simulation du processus de Hawkes
    simu = SimuHawkesExpKernels([[alpha]], beta, [mu], end_time)

    # Simulation du processus de Hawkes
    simu.simulate()

    # Filtration des predictions pour ne garder que les evenements futurs
    future_events = [t for t in simu.timestamps[0] if t >= start_time]

    # Conversion en tableau numpy et arrondi a l'unite la plus proche
    future_events = np.rint(np.array(future_events)).astype(int)

    return future_event

