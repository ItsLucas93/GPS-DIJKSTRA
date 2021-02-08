# ----------- Bibliothèque ------------
import csv
from itertools import islice
import math
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# Fichier data pour métro
from data import lignes_index, long, lat
from data import station as station_data

# Plt parameters
plt.rcParams['axes.facecolor'] = 'grey'


# ------------ KPP ----------------

def RADIANS(x):
    return x * math.pi / 180.


def distance_entre_deux_Points_GPS(long_A, lat_A, long_B, lat_B):
    return round((1000 * R_terre * math.acos(
        math.sin(RADIANS(long_B)) * math.sin(RADIANS(long_A)) + math.cos(RADIANS(long_B)) * math.cos(
            RADIANS(long_A)) * math.cos(RADIANS(lat_A - lat_B)))))


def eleve_distance(all_eleve, all_station):
    resultat = ['distance en m']
    long_A = [l[1][1] for l in all_eleve]
    lat_A = [l[1][0] for l in all_eleve]
    long_B = [s[2][1] for s in all_station]
    lat_B = [s[2][0] for s in all_station]
    for eleve in range(len(all_eleve)):
        for station in range(len(all_station)):
            if all_eleve[eleve][2] == all_station[station][0]:
                resultat.append(
                    distance_entre_deux_Points_GPS(long_A[eleve], lat_A[eleve], long_B[station], lat_B[station]))
    return resultat


def insert_distance_station(data, csv_file):
    num = 0
    with open(csv_file) as csvfile:
        rows = [row for row in csv.reader(csvfile)]
        with open(csv_file, 'w', encoding='utf-8', newline='') as f:
            f.truncate()
            writer = csv.writer(f)
            for row in rows:
                row.append(data[num])
                writer.writerow(row)
                num += 1

def kpp(trainData, testData, labels, k):
    distances_voisins = []
    for index, sample in enumerate(trainData):
        distance = distance_euclidienne(sample, testData)
        distances_voisins.append((distance, index))
    distances_voisins = sorted(distances_voisins)
    idices_des_k_voisins = [index for distance, index in distances_voisins[:k]]
    return labels[idices_des_k_voisins[0]]


def distance_euclidienne(p1, p2):
    distance_carre = 0
    for i in range(len(p1)):
        distance_carre = distance_carre + (p1[i] - p2[i]) ** 2
    return math.sqrt(distance_carre)


def classifieur(trainData, testData, k):
    labels = [data[2] for data in trainData]
    position = [data[1] for data in trainData]
    postition_test = []
    for data in testData:
        postition_test.append(data[1])
    resultat = []
    for data in range(len(postition_test)):
        resultat.append([testData[data][0], testData[data][1], kpp(position, postition_test[data], labels, k)])
    for data in trainData:
        resultat.append(data)
    return resultat  # [noms [latitude,longitude,station],[...],...]


def getAccuracy(testData, prediction):
    correct = 0
    for i in range(len(testData)):
        if testData[i] == prediction[i]:
            correct += 1
    return (correct / float(len(testData))) * 100


def determiner_echantilon(base, k, echantilon):
    position_d1 = [p[1] for p in base]
    station_d1 = [station[2] for station in base]
    resultat = [kpp(position_d1, e[1], station_d1, k) for e in echantilon]
    return resultat


def dessiner(base, echantilon=[]):
    v1, v2, v3, v4 = 0, 0, 0, 0
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(221)
    ax1.set_title('Scatter Plot1')
    plt.xlabel('X')
    plt.ylabel('Y')
    station = list(set([s[2] for s in base]))
    for data in base:
        if v1 == 0 and data[2] == station[0]:
            v1 += 1
            ax1.scatter(data[1][0], data[1][1], c='r', marker='o', label='vaneau')
        elif v1 != 0 and data[2] == station[0]:
            ax1.scatter(data[1][0], data[1][1], c='r', marker='o')

        elif v2 == 0 and data[2] == station[1]:
            v2 += 1
            ax1.scatter(data[1][0], data[1][1], c='b', marker='o', label='sevre')
        elif v2 != 0 and data[2] == station[1]:
            ax1.scatter(data[1][0], data[1][1], c='b', marker='o')

        elif v3 == 0 and data[2] == station[2]:
            v3 += 1
            ax1.scatter(data[1][0], data[1][1], c='g', marker='o', label='duroc')
        elif v3 != 0 and data[2] == station[2]:
            ax1.scatter(data[1][0], data[1][1], c='g', marker='o')

    if echantilon != []:
        for data in echantilon:
            if v4 == 0:
                v4 += 1
                ax1.scatter(data[1][0], data[1][1], c='black', marker='x', label='non défini')
            else:
                ax1.scatter(data[1][0], data[1][1], c='black', marker='x')
    plt.legend()
    plt.show()


def Insert_echantilon(nllg, csv_file):
    nllgcopy = [[n[0], n[1][0], n[1][1], n[2]] for n in nllg]
    with open(csv_file, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for data in nllgcopy:
            writer.writerow(data)




# ------------- Métro --------------

class trajet():
    def __init__(self, cart, base):
        self.time = 1.5  # min
        self.cart = cart
        self.all_station = []
        for ligne in range(len(base)):
            self.all_station.append(base[ligne][0])
        self.all_station_solo = [base[num][0] for num in range(len(base))]
        self.base = base

    def construction_graph_matrice(self):
        graph = {}
        for x in self.cart:
            for i in range(len(x)):
                try:
                    if x[i] in graph:
                        graph[x[i]].append(x[i - 1])
                        graph[x[i]].append(x[i + 1])
                    elif i - 1 <= -1:
                        if x[i] in graph:
                            graph[x[i]].append(x[i + 1])
                        else:
                            graph[x[i]] = [x[i + 1]]
                    else:
                        graph[x[i]] = [x[i - 1], x[i + 1]]
                except IndexError:
                    if x[i] in graph:
                        graph[x[i]].append(x[i - 1])
                    else:
                        graph[x[i]] = [x[i - 1]]
        return graph

    def construction_graph_dict(self):
        graph = {}
        for x in self.cart:
            for i in range(len(x)):
                try:
                    if i - 1 <= -1:
                        if x[i] in graph:
                            graph[x[i]].update({x[i + 1]: 1})
                        else:
                            graph[x[i]] = {x[i + 1]: 1}
                    elif x[i] in graph:
                        graph[x[i]].update({x[i - 1]: 1})
                        graph[x[i]].update({x[i + 1]: 1})
                    else:
                        graph[x[i]] = {x[i - 1]: 1}
                        graph[x[i]].update({x[i + 1]: 1})
                except IndexError:
                    if x[i] in graph:
                        graph[x[i]].update({x[i - 1]: 1})
                    else:
                        graph[x[i]] = {x[i - 1]: 1}
        return graph

    def matrice(self):
        graph = self.construction_graph_matrice()
        matrice = [[] for i in range(len(self.all_station))]
        for num in range(len(self.all_station)):
            for nume in range(len(self.all_station)):
                if self.all_station[nume] in graph[self.all_station[num]]:
                    matrice[num].append(1)
                else:
                    matrice[num].append(0)
        return matrice

    def dijkstra_matrice(self, start):
        resultat = []
        mg = self.matrice()
        s = self.all_station.index(start)
        infini = sum(sum(ligne) for ligne in mg) + 1
        nb_sommet = len(mg)

        s_connu = {s: [0, [self.all_station[s]]]}
        s_inconnu = {k: [infini, ''] for k in range(nb_sommet) if k != s}

        for suivant in range(nb_sommet):
            if mg[s][suivant]:
                s_inconnu[suivant] = [mg[s][suivant], s]

        while s_inconnu and any(s_inconnu[k][0] < infini for k in s_inconnu):
            u = min(s_inconnu, key=s_inconnu.get)
            longueur_u, precedent_u = s_inconnu[u]
            for v in range(nb_sommet):
                if mg[u][v] and v in s_inconnu:
                    d = longueur_u + mg[u][v]
                    if d < s_inconnu[v][0]:
                        s_inconnu[v] = [d, u]
            s_connu[u] = [longueur_u, s_connu[precedent_u][1] + [self.all_station[u]]]
            del s_inconnu[u]
            # print('longuer',longueur_u,':','-'.join(map(str,s_connu[u][1])))
            resultat.append(s_connu[u][1])

        # for k in s_inconnu:
        # print('il')
        return resultat.pop()

    def dijkstra(self, s):
        G = self.construction_graph_dict()
        inf = sum(sum(G[sommet][i] for i in G[sommet]) for sommet in G) + 1
        global s_explore
        global s_a_explorer
        s_explore = {s: [0, [s]]}
        s_a_explorer = {j: [inf, ""] for j in G if j != s}
        for suivant in G[s]:
            s_a_explorer[suivant] = [G[s][suivant], s]
        while s_a_explorer and any(s_a_explorer[k][0] < inf for k in s_a_explorer):
            s_min = min(s_a_explorer, key=s_a_explorer.get)
            longueur_s_min, precedent_s_min = s_a_explorer[s_min]
            for successeur in G[s_min]:
                if successeur in s_a_explorer:
                    dist = longueur_s_min + G[s_min][successeur]
                    if dist < s_a_explorer[successeur][0]:
                        s_a_explorer[successeur] = [dist, s_min]
            s_explore[s_min] = [longueur_s_min, s_explore[precedent_s_min][1] + [s_min]]
            del s_a_explorer[s_min]

        return s_explore

    # --------------------- Entrée et sortie du graphe ------------------

    def entree_sortie(self, entree, sortie):
        self.dijkstra(entree)
        for k in s_explore:
            if sortie == k:
                return (s_explore[k][1])

        for k in s_a_explorer:
            if sortie == k:
                return ("Il n\'existe aucun chemin de {} à {}".format(entree, sortie))

    def distance_entre_station(self):
        station_position = {}
        # {nom_station:[latitude,longitude]}
        for num in range(len(self.all_station_solo)):
            station_position[self.all_station_solo[num]] = [self.base[num][2][0], self.base[num][2][1]]
        print(station_position)
        return station_position

    def matrice_distance(self):
        station_position = self.distance_entre_station()
        graph = self.construction_graph_matrice()
        matrice_distance = [[] for i in range(len(self.all_station))]
        for num in range(len(self.all_station)):
            for nume in range(len(self.all_station)):
                if self.all_station[nume] in graph[self.all_station[num]]:
                    matrice_distance[num].append(
                        distance_entre_deux_Points_GPS(station_position[self.all_station[num]][1],
                                                       station_position[self.all_station[num]][0],
                                                       station_position[self.all_station[nume]][1],
                                                       station_position[self.all_station[nume]][0]))
                else:
                    matrice_distance[num].append(0)
        return matrice_distance


def station_plus_proche(longitude, latitude, sap):
    resultat = [distance_entre_deux_Points_GPS(longitude, latitude, p[2][1], p[2][0]) for p in sap]
    print(sap[resultat.index(min(resultat))][0])
    return sap[resultat.index(min(resultat))][0]


def plot1(longitude, latitude, sap, all_station, sortie):
    station_coord = {}
    for i in sap:
        station_coord[i[0]] = i[2]
    depart = station_plus_proche(longitude, latitude, sap)
    dde = trajet(all_station, sap)
    chemin = dde.entree_sortie(depart, sortie)
    longitude_station = [sap[dde.all_station.index(station)][2][1] for station in chemin]
    latitude_station = [sap[dde.all_station.index(station)][2][0] for station in chemin]
    for i in all_station:
        longitude = [station_coord[i[l]][1] for l in range(len(i))]
        latitude = [station_coord[i[l]][0] for l in range(len(i))]

    couleurs_metro = ["", "gold", "", "", "darkviolet", "", "", "", "blueviolet", "", "saddlebrown", "", "darkgreen",
                      "paleturquoise"]
    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    plt.figure(figsize=(13, 11))
    plt.title(chemin)
    x = long
    y = lat
    nom = station_data
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
    plt.scatter(longitude, latitude, label="Votre position", alpha=0.5, zorder=2, marker="P", c='red')
    plt.plot(longitude_station, latitude_station, label='Itinéraire', c='white', marker="o", markersize=8)
    plt.legend(loc='upper right')
    plt.savefig("./img/plot1.png")
    # plt.show()


def nb_autour(longitude, latitude, d1, rayon):
    les_autres = [[x[1][0], x[1][1]] for x in d1]
    labels = [x[2] for x in d1]
    distances_voisins = []
    for index, sample in enumerate(les_autres):
        distance = distance_entre_deux_Points_GPS(sample[1], sample[0], longitude, latitude)
        distances_voisins.append((distance, index))
    distances_voisins = sorted(distances_voisins)
    idices_des_k_voisins = [labels[index] for distance, index in distances_voisins if distance <= rayon]
    station = []
    for s in idices_des_k_voisins:
        if s not in station:
            station.append(s)
    eleve_proche = {}
    for s in station:
        name = Counter()
        for num in idices_des_k_voisins:
            name[num] += 1
        eleve_proche[s] = name[s]
    resultat = {}
    for name, nb in eleve_proche.items():
        resultat[name] = nb / len(idices_des_k_voisins)
    return resultat, len(idices_des_k_voisins), distances_voisins


def plot2(longitude, latitude, d1, rayon):
    rayon1 = rayon / 100000
    nb = nb_autour(longitude, latitude, d1, rayon)
    v1, v2, v3 = 0, 0, 0
    fig = plt.figure(figsize=(13, 13))
    ax1 = fig.add_subplot(221)
    station = list(set([s[2] for s in d1]))
    for data in d1:
        if v1 == 0 and data[2] == station[0]:
            v1 += 1
            ax1.scatter(data[1][0], data[1][1], c='r', marker='o', label='Vaneau')
        elif v1 != 0 and data[2] == station[0]:
            ax1.scatter(data[1][0], data[1][1], c='r', marker='o')

        elif v2 == 0 and data[2] == station[1]:
            v2 += 1
            ax1.scatter(data[1][0], data[1][1], c='b', marker='o', label='Sevre')
        elif v2 != 0 and data[2] == station[1]:
            ax1.scatter(data[1][0], data[1][1], c='b', marker='o')

        elif v3 == 0 and data[2] == station[2]:
            v3 += 1
            ax1.scatter(data[1][0], data[1][1], c='g', marker='o', label='Duroc')
        elif v3 != 0 and data[2] == station[2]:
            ax1.scatter(data[1][0], data[1][1], c='g', marker='o')

    x = np.linspace(latitude - rayon1, latitude + rayon1, 5000)
    y1 = np.sqrt(rayon1 ** 2 - (x - latitude) ** 2) + longitude
    y2 = -np.sqrt(rayon1 ** 2 - (x - latitude) ** 2) + longitude
    plt.plot(x, y1, c='k')
    plt.plot(x, y2, c='k')
    plt.xticks([])
    plt.yticks([])
    ax1.scatter(latitude, longitude, c='black', label="Dans {}m il y a {} élèves".format(rayon, nb[1]))
    for name, nb in nb[0].items():
        ax1.scatter(latitude, longitude, c='black', label="{} est {}%".format(name, nb * 100))
    plt.legend(loc='upper left')
    plt.savefig("./img/plot2.png")


def plot3(longitude, latitude, d1, rayon):
    rayon1 = rayon / 100000
    nb = nb_autour(longitude, latitude, d1, rayon)
    v1, v2, v3 = 0, 0, 0
    fig = plt.figure(figsize=(17, 17))
    ax1 = fig.add_subplot(221)
    station = list(set([s[2] for s in d1]))
    for data in d1:
        if v1 == 0 and data[2] == station[0]:
            v1 += 1
            ax1.scatter(data[1][0], data[1][1], c='r', marker='o', label='Vaneau')
        elif v1 != 0 and data[2] == station[0]:
            ax1.scatter(data[1][0], data[1][1], c='r', marker='o')

        elif v2 == 0 and data[2] == station[1]:
            v2 += 1
            ax1.scatter(data[1][0], data[1][1], c='b', marker='o', label='Sevre')
        elif v2 != 0 and data[2] == station[1]:
            ax1.scatter(data[1][0], data[1][1], c='b', marker='o')

        elif v3 == 0 and data[2] == station[2]:
            v3 += 1
            ax1.scatter(data[1][0], data[1][1], c='g', marker='o', label='Duroc')
        elif v3 != 0 and data[2] == station[2]:
            ax1.scatter(data[1][0], data[1][1], c='g', marker='o')

    x = np.linspace(latitude - rayon1, latitude + rayon1, 5000)
    y1 = np.sqrt(rayon1 ** 2 - (x - latitude) ** 2) + longitude
    y2 = -np.sqrt(rayon1 ** 2 - (x - latitude) ** 2) + longitude
    plt.plot(x, y1, c='k')
    plt.plot(x, y2, c='k')
    plt.xticks([])
    plt.yticks([])
    distances_voisins = []
    for index, sample in enumerate(d1):
        distance = distance_euclidienne(sample[1], [longitude, latitude])
        distances_voisins.append((distance, index))
    distances_voisins = sorted(distances_voisins)
    idices_des_k_voisins = [index for distance, index in distances_voisins]
    print(d1[idices_des_k_voisins[0]][1][1], d1[idices_des_k_voisins[0]][1][0])
    ax1.scatter(latitude, longitude, c='black', label="Dans {}m il y a {} élèves".format(rayon, nb[1]))
    plt.annotate('{} , {}'.format(d1[idices_des_k_voisins[-1]][0],
                                  distance_entre_deux_Points_GPS(d1[idices_des_k_voisins[-1]][1][1],
                                                                 d1[idices_des_k_voisins[-1]][1][0], longitude,
                                                                 latitude)),
                 xy=(d1[idices_des_k_voisins[-1]][1][0], d1[idices_des_k_voisins[-1]][1][1]))
    for name, nb in nb[0].items():
        ax1.scatter(latitude, longitude, c='black', label="{} est {}%".format(name, nb * 100))
    plt.legend(loc='upper left')
    plt.savefig("./img/plot3.png")


if __name__ == "__main__":

    # ---------------- Exercice KPP ----------------

    csvfile = open(r'./csv/data_eleve_GPS_Groupe.csv', 'r')
    readCSV = csv.reader(csvfile)
    nom_p_station = [[data[0], [float(data[1]), float(data[2])], data[3]] for data in
                     islice(readCSV, 1, None)]

    p = [p[1] for p in nom_p_station]
    station = [s[2] for s in nom_p_station]
    fanny = [48.8468570444776, 2.31573327790118]

    p_test_eleve = [1, 2, 3, 10, 11, 19, 20, 21]
    position_test = [nom_p_station[num][1] for num in p_test_eleve]
    station_test = [nom_p_station[num][2] for num in p_test_eleve]
    nom_p_station_test = [nom_p_station[num] for num in p_test_eleve]
    nom_p_apprentissage = [[p[0], p[1]] for p in nom_p_station]
    station_apprentissage = [s[2] for s in nom_p_station]

    for num in p_test_eleve[::-1]:
        del nom_p_apprentissage[num]
        del station_apprentissage[num]

    k = 2
    # print(kpp(p,fanny,station,k))
    # print(kpp(position_test,position_test[2],station_test,k))
    d1 = classifieur(nom_p_station_test, nom_p_apprentissage, k)
    # print(d1)
    # print(getAccuracy(station_apprentissage,[d1[n][2] for n in range(len(nom_p_apprentissage))]))

    # dessiner(d1)

    csvfile = open(r'./csv/echantilon_eleve.csv', 'r')
    readCSV = csv.reader(csvfile)
    nom_p_echantilon = [[data[0], [float(data[2]), float(data[1])]] for data in readCSV]
    # dessiner(d1,nom_p_echantilon)

    echantilon = determiner_echantilon(d1, k, nom_p_echantilon)
    echantilon = [nom_p_echantilon[s].append(echantilon[s]) for s in range(len(echantilon))]

    d2 = classifieur(d1, nom_p_echantilon, k)
    # dessiner(d2)

    # Insert_echantilon(nom_p_echantilon,"data_eleve_GPS_Groupe.csv")
    R_terre = 6372.954775981000

    # --------------- Métro ---------------------

    csvfile = open(r'./csv/station.csv', 'r')
    readCSV = csv.reader(csvfile)
    station_adresse_position = [[data[0], data[1], [float(data[3]), float(data[2])]] for data in
                                islice(readCSV, 1, None)]  # nom de station,adresse,[latitude,longitude]
    sap = station_adresse_position

    distance_eleve = eleve_distance(d2, station_adresse_position)
    # insert_distance_station(distance_eleve,"data_eleve_GPS_Groupe.csv")

    ligne10 = [sap[s][0] for s in range(5)]
    ligne4 = [sap[s][0] for s in range(4, 7)]
    ligne1 = [sap[s][0] for s in range(6, 11)]
    ligne12 = [sap[s][0] for s in range(10, 14)]
    ligne12.append('Sevres')
    ligne8 = ['INVALIDE', 'Concorde']
    ligne13 = ['DUROC', 'SAINT-FRANCOIS-XAVIER', 'Varenne', 'INVALIDE']
    all_station = [ligne10, ligne4, ligne1, ligne12, ligne8, ligne13]
    dde = trajet(all_station, sap)
    # dde.matrice_distance()
    # print(dde.construction_graph_dict())
    chemin = dde.entree_sortie('DUROC', 'SOLFERINO-BELLECHASSE')

    # ---------------------- PHP ---------------------

    csvfile = open(r'./csv/coordonnee.csv', 'r')
    readCSV = csv.reader(csvfile)
    coordonnee = [x for x in readCSV]
    coordonnee = coordonnee[0]
    page = coordonnee[3]

    if page == 'page1':
        coordonnee = [float(coordonnee[0]), float(coordonnee[1]), coordonnee[2]]
        csvfile.close()
        plot1(coordonnee[0], coordonnee[1], sap, all_station, coordonnee[2])
    elif page == 'page2':
        coordonnee = [float(coordonnee[0]), float(coordonnee[1]), float(coordonnee[2])]
        csvfile.close()
        plot2(coordonnee[0], coordonnee[1], d1, coordonnee[2])
    elif page == 'page3':
        coordonnee = [float(coordonnee[0]), float(coordonnee[1]), float(coordonnee[2])]
        csvfile.close()
        plot3(coordonnee[0], coordonnee[1], d1, coordonnee[2])
    csvfile.close()
