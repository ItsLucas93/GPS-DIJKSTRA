"""
Auteur: Lucas
Classe: T-7
Date: 01/02/2021
"""

# ----------- Bibliothèque ------------

from math import exp, cos, sin, acos, pi
import csv
from itertools import islice
import numpy as np
import matplotlib.pyplot as plt

# Plt parameters
plt.rcParams['axes.facecolor'] = 'grey'

# --------------- Métro ---------------

correspondances_station = {"Duroc": ["Vaneau", "Saint-François-Xavier"], "Vaneau": ["Duroc", "Sevres"],
                           "Sevres": ["Vaneau", "Mabillon", "Rue du bac"], "Mabillon": ["Odeon", "Sevres"],
                           "Odeon": ["Mabillon", "Cite"], "Cite": ["Chatelet-Les-Halles", "Odeon"],
                           "Chatelet-Les-Halles": ["Cite", "Louvre-Rivoli"],
                           "Louvre-Rivoli": ["Chatelet-Les-Halles", "Palais-Royal"],
                           "Palais-Royal": ["Tuileries", "Louvre-Rivoli"], "Tuileries": ["Concorde", "Palais-Royal"],
                           "Concorde": ["Tuileries", "Invalide", "Assemblee Nationale"],
                           "Assemblee Nationale": ["Concorde", "Solferino-Bellechase"],
                           "Solferino-Bellechase": ["Assemblee Nationale", "Rue du bac"],
                           "Rue du bac": ["Solferino-Bellechase", "Sevres"],
                           "Varenne": ["Invalide", "Saint-François-Xavier"], "Invalide": ["Concorde", "Varenne"],
                           "Saint-François-Xavier": ["Duroc", "Varenne"]}
lignes_metro = {1: ["Chatelet-Les-Halles", "Louvre-Rivoli", "Palais-Royal", "Tuileries", "Concorde"],
                4: ["Chatelet-Les-Halles", "Cite", "Odeon"],
                8: ["Concorde", "Varenne", "Invalide"],
                10: ["Duroc", "Vaneau", "Sevres", "Mabillon", "Odeon"],
                12: ["Concorde", "Assemblee Nationale", "Solferino-Bellechase", "Rue du bac", "Sevres"],
                13: ["Duroc", "Saint-François-Xavier", "Varenne", "Invalide"]}
lignes_index = {1: [6, 7, 8, 9, 10],
                4: [6, 5, 4],
                8: [10, 15, 14], 10: [0, 1, 2, 3, 4],
                12: [10, 11, 12, 13, 2],
                13: [0, 16, 14, 15]}

# ----------- Donées CSV --------------

csvfile = open('./csv/station.csv', 'r')
readCSV = csv.reader(csvfile)

station_csv = []
for data in islice(readCSV, 1, None):
    station_csv.append([data[0], float(data[2]), float(data[3])])

long = [p[1] for p in station_csv]
lat = [p[2] for p in station_csv]
station = [s[0] for s in station_csv]

# -------------- Fonctions ---------------

def RADIANS(x):
    return x * pi / 180.


R_terre = 6372.954775981000


def distance_entre_deux_Points_GPS(long_A, lat_A, long_B, lat_B):
    """donne la distance entre deux points en M"""
    return (1000 * R_terre * acos(
        sin(RADIANS(long_B)) * sin(RADIANS(long_A)) + cos(RADIANS(long_B)) * cos(RADIANS(long_A)) * cos(
            RADIANS(lat_A - lat_B))))


d = distance_entre_deux_Points_GPS(2.31378950871346, 48.8516025709589, 2.31643327790118, 48.8473070444776)


def gen_matrice(dicho):
    # Exploitation CSV
    csvfile = open('./csv/station.csv', 'r')
    readCSV = csv.reader(csvfile)

    station_csv = []
    for data in islice(readCSV, 1, None):
        station_csv.append([data[0], data[1], float(data[2]), float(data[3])])
    long = [p[2] for p in station_csv]
    lat = [s[3] for s in station_csv]
    # Retrait des labels
    long.pop(0)
    lat.pop(0)

    listedele = [key for key in dicho]
    n = len(listedele)

    # Création d'une matrice à partir avec only 0
    matrice = np.zeros((n, n))

    # Ajout des valeurs dans la matrice depuis le dico
    for key in dicho:
        val = dicho[key]
        clef = listedele.index(key)
        for i in val:
            blo = listedele.index(i)
            matrice[clef][blo] = distance_entre_deux_Points_GPS(float(long[clef - 1]), float(lat[clef - 1]),
                                                                float(long[blo - 1]), float(lat[blo - 1]))
    return matrice, listedele


def dijkstra_matrice(M, s):
    inf = sum(sum(ligne) for ligne in M) + 1
    nb_sommets = len(M)
    s_explore = {s: [0, [s]]}
    s_a_explorer = {j: [inf, ""] for j in range(nb_sommets) if j != s}
    for suivant in range(nb_sommets):
        if M[s][suivant]:
            s_a_explorer[suivant] = [M[s][suivant], s]

    print("Dans le graphe d\'origine {} de matrice d\'adjacence :".format(s))

    while s_a_explorer and any(s_a_explorer[k][0] < inf for k in s_a_explorer):
        s_min = min(s_a_explorer, key=s_a_explorer.get)
        longueur_s_min, precedent_s_min = s_a_explorer[s_min]
        for successeur in range(nb_sommets):
            if M[s_min][successeur] and successeur in s_a_explorer:
                dist = longueur_s_min + M[s_min][successeur]
                if dist < s_a_explorer[successeur][0]:
                    s_a_explorer[successeur] = [dist, s_min]
        s_explore[s_min] = [longueur_s_min, s_explore[precedent_s_min][1] + [s_min]]
        del s_a_explorer[s_min]

    for k in s_a_explorer:
        print("Il n\'y a pas de chemin de {} à {}".format(s, k))

    return s_explore, s_a_explorer


def entree_sortie(M, entree, sortie, dicho):
    listedele = [key for key in dicho]
    s_explore, s_a_explorer = dijkstra_matrice(M, int(entree))
    for k in s_explore:
        if int(sortie) == k:
            global path
            path = [listdele[i] for i in s_explore[k][1]]
            entree2 = int(entree)
            sortie2 = int(sortie)
            print("Le plus court chemin menant de {} à {} est {} ".format(listdele[entree2], listdele[sortie2],
                                                                          path))
            print("Son poids est égal à {}".format(s_explore[k][0]))
            return s_explore[k][0], s_explore[k][1]

    for k in s_a_explorer:
        if sortie == k:
            print("Il n\'existe aucun chemin de {} à {}".format(entree, sortie))


def tracage_metro():
    couleurs_metro = ["", "gold", "", "", "darkviolet", "", "", "", "blueviolet", "", "saddlebrown", "", "darkgreen",
                      "paleturquoise"]
    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    x = long
    y = lat
    nom = station
    for i in range(len(x)):
        plt.text(x[i], y[i], nom[i])
    for key in lignes_index:
        label = "ligne" + str(key)
        color = int(key)
        val = lignes_index[key]
        x = [long[y] for y in val]
        y = [lat[y] for y in val]

        plt.plot(x, y, "b:o", label=label, alpha=0.8, zorder=2, color=couleurs_metro[color])
        plt.scatter(x, y, alpha=0.5, zorder=3, color=couleurs_metro[color])

    long2 = [long[y] for y in liste_station_nbr]
    lat2 = [lat[y] for y in liste_station_nbr]
    plt.plot(long2, lat2, alpha=2, zorder=5, color='white', label='parcours')
    plt.legend()


def enregistre():
    plt.savefig('foo.png')


entree = int(input("Quel est votre sommet d'entrée ? "))
sortie = int(input("Quel est votre sommet de sortie ? "))

matrice, listdele = gen_matrice(correspondances_station)
distance, liste_station_nbr = entree_sortie(matrice, entree, sortie, correspondances_station)

# tracage_metro()
# enregistre()
# plt.show()






