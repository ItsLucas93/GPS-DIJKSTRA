"""
Auteur: Lucas
Classe: T-7
Date: 01/02/2021
"""

# ----------- Bibliothèque ------------

from math import exp, cos, sin, acos, pi
import math
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

# ----------- Donées (CSV + Autres) --------------

csvfile = open('../csv/station.csv', 'r')
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
    csvfile = open('../csv/station.csv', 'r')
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

def dijkstra_graphe(self, s):
    G=self.graph_construit_2()
    inf = sum(sum(G[sommet][i] for i in G[sommet]) for sommet in G) + 1
    global s_explore
    global s_a_explorer
    s_explore = {s : [0, [s]]}
    s_a_explorer = {j : [inf, ""] for j in G if j != s}
    for suivant in G[s]:
        s_a_explorer[suivant] = [G[s][suivant], s]
    while s_a_explorer and any(s_a_explorer[k][0] < inf for k in s_a_explorer):
        s_min = min(s_a_explorer, key = s_a_explorer.get)
        longueur_s_min, precedent_s_min = s_a_explorer[s_min]
        for successeur in G[s_min]:
            if successeur in s_a_explorer:
                dist = longueur_s_min + G[s_min][successeur]
                if dist < s_a_explorer[successeur][0]:
                    s_a_explorer[successeur] = [dist, s_min]
        s_explore[s_min] = [longueur_s_min, s_explore[precedent_s_min][1] + [s_min]]
        del s_a_explorer[s_min]

    return s_explore


def dijkstra_matrice(matrice, s):
    inf = sum(sum(ligne) for ligne in matrice) + 1
    nb_sommets = len(matrice)
    s_explore = {s: [0, [s]]}
    s_a_explorer = {j: [inf, ""] for j in range(nb_sommets) if j != s}
    for suivant in range(nb_sommets):
        if matrice[s][suivant]:
            s_a_explorer[suivant] = [matrice[s][suivant], s]

    print("Dans le graphe d\'origine {} de matrice d\'adjacence :".format(s))

    while s_a_explorer and any(s_a_explorer[k][0] < inf for k in s_a_explorer):
        s_min = min(s_a_explorer, key=s_a_explorer.get)
        longueur_s_min, precedent_s_min = s_a_explorer[s_min]
        for successeur in range(nb_sommets):
            if matrice[s_min][successeur] and successeur in s_a_explorer:
                dist = longueur_s_min + matrice[s_min][successeur]
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
    plt.figure(figsize=(13, 11))
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
    plt.savefig('../img/plot1.png')


# # entree = int(input("Quel est votre sommet d'entrée ? "))
# # sortie = int(input("Quel est votre sommet de sortie ? "))
# entree = 1
# sortie = 4

# matrice, listdele = gen_matrice(correspondances_station)
# distance, liste_station_nbr = entree_sortie(matrice, entree, sortie, correspondances_station)

# tracage_metro()
# enregistre()
# plt.show()

# ------------------ KPP et Classifieur ------------------

def kpp(trainData, testData, labels, k=3):
    distances_voisins=[]
    for index,sample in enumerate(trainData):
        distance = distance_euclidienne(sample,testData)
        distances_voisins.append((distance,index))
    distances_voisins=sorted(distances_voisins)
    idices_des_k_voisins=[index for distance,index in distances_voisins[:k]]
    return labels[idices_des_k_voisins[0]]

def distance_euclidienne(p1,p2):
    distance_carre=0
    for i in range(len(p1)):
        distance_carre=distance_carre+(p1[i]-p2[i])**2
    return math.sqrt(distance_carre)

def classifieur(trainData, testData,k):
    labels=[donnee[2] for donnee in trainData]#station
    position=[donnee[1] for donnee in trainData]#latitude longitude
    postition_test=[]
    for donnee in testData:
        postition_test.append(donnee[1])
    resultat=[]
    for donnee in range(len(postition_test)):
        resultat.append([testData[donnee][0],testData[donnee][1],kpp(position, postition_test[donnee],labels , k)])
    for donnee in trainData:
        resultat.append(donnee)
    return resultat #[noms [latitude,longitude,station],[...],...]

def getAccuracy(testData,prediction):
    correct=0
    for i in range(len(testData)):
        if testData[i]==prediction[i]:
            correct+=1
    return (correct/float(len(testData))) *100

def determiner_echantilon(base,k,echantilon):
    position_d1=[p[1] for p in base]
    station_d1=[station[2] for station in base]
    resultat=[kpp(position_d1,e[1],station_d1,k) for e in echantilon]
    return resultat

def dessiner(base,echantilon=[]):
    v1,v2,v3,v4=0,0,0,0
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(221)
    ax1.set_title('Scatter Plot1')
    plt.xlabel('X')
    plt.ylabel('Y')
    station=list(set([s[2] for s in base]))
    for donnee in base:
        if v1==0 and donnee[2]==station[0]:
            v1+=1
            ax1.scatter(donnee[1][0],donnee[1][1],c = 'r',marker = 'o',label='vaneau')
        elif v1!=0 and donnee[2]==station[0]:
            ax1.scatter(donnee[1][0],donnee[1][1],c = 'r',marker = 'o')
            
        elif v2==0 and donnee[2]==station[1]:
            v2+=1
            ax1.scatter(donnee[1][0],donnee[1][1],c = 'b',marker = 'o',label='sevre') 
        elif v2!=0 and donnee[2]==station[1]:
            ax1.scatter(donnee[1][0],donnee[1][1],c = 'b',marker = 'o')
            
        elif v3==0 and donnee[2]==station[2]:
            v3+=1
            ax1.scatter(donnee[1][0],donnee[1][1],c = 'g',marker = 'o',label='duroc') 
        elif v3!=0 and donnee[2]==station[2]:
            ax1.scatter(donnee[1][0],donnee[1][1],c = 'g',marker = 'o')

    if echantilon!=[]:
        for donnee in echantilon:
            if v4 == 0:
                v4+=1
                ax1.scatter(donnee[1][0],donnee[1][1],c = 'black',marker = 'x',label='non défini') 
            else:
                ax1.scatter(donnee[1][0],donnee[1][1],c = 'black',marker = 'x')
    plt.legend()
    plt.show()
        

def Insert_echantilon(nllg,nom_csv):
    nllgcopy=[[n[0],n[1][0],n[1][1],n[2]] for n in nllg]
    with open(nom_csv, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for donnee in nllgcopy:
            writer.writerow(donnee)


if __name__ == "__main__":

    csvfile = open('../csv/data_eleve_GPS_Groupe.csv','r')
    readCSV = csv.reader(csvfile)
    nom_station_p=[[donnee[0],[float(donnee[1]),float(donnee[2])],donnee[3]] for donnee in islice(readCSV, 1, None)]

    p=[p[1] for p in nom_station_p]
    station=[s[2] for s in nom_station_p]
    fanny=[48.8468570444776,2.31573327790118]

    p_test_eleve=[1,2,3,10,11,19,20,21]
    position_test=[nom_station_p[num][1] for num in p_test_eleve]
    station_test=[nom_station_p[num][2] for num in p_test_eleve]
    nom_station_test=[nom_station_p[num] for num in p_test_eleve]
    train_data=[[p[0],p[1]] for p in nom_station_p]
    station_train=[s[2] for s in nom_station_p]

    for num in p_test_eleve[::-1]:
        del train_data[num]
        del station_train[num]
        
    k = 2
    #print(kpp(p,fanny,station,k))
    #print(kpp(position_test,position_test[2],station_test,k))
    d1 = classifieur(nom_station_test,train_data,k)
    #print(d1)
    #print(getAccuracy(station_train,[d1[n][2] for n in range(len(train_data))]))

    #dessiner(d1)

    csvfile = open('../csv/echantilon_eleve.csv','r')
    readCSV = csv.reader(csvfile)
    nom_p_echantilon=[[donnee[0],[float(donnee[2]),float(donnee[1])]] for donnee in readCSV]
    #dessiner(d1,nom_p_echantilon)


    echantilon=determiner_echantilon(d1,k,nom_p_echantilon)
    echantilon=[nom_p_echantilon[s].append(echantilon[s]) for s in range(len(echantilon))]

    d2=classifieur(d1,nom_p_echantilon,k)
    #dessiner(d2)

    #Insert_echantilon(nom_p_echantilon,"data_eleve_GPS_Groupe.csv")
    
    csvfile = open('../csv/station.csv','r')
    readCSV = csv.reader(csvfile)
    station_adresse_position = [[donnee[0],donnee[1],[float(donnee[3]),float(donnee[2])]] for donnee in islice(readCSV, 1, None)]#nom de station,adresse,[latitude,longitude]
    sap = station_adresse_position

    #insert_distance_station(distance_eleve,"data_eleve_GPS_Groupe.csv")

    #dde.matrice_distance()
    #print(dde.graph_construit_2())

    # csvfile = open('../csv/coordonnee.csv','r')
    readCSV = csv.reader(csvfile)
    coordonnee=[x for x in readCSV]
    coordonnee=coordonnee[0]
    page=coordonnee[3]
    if page =='page1':
        coordonnee=[float(coordonnee[0]),float(coordonnee[1]),coordonnee[2]]
        csvfile.close()
        plot_station(coordonnee[0],coordonnee[1],sap,all_station,coordonnee[2])
    elif page == 'page2':
        coordonnee=[float(coordonnee[0]),float(coordonnee[1]),float(coordonnee[2])]
        csvfile.close()
        print("qsdqs2")
    elif page == 'page3':
        coordonnee=[float(coordonnee[0]),float(coordonnee[1]),float(coordonnee[2])]
        csvfile.close()
        plot_rayon1(coordonnee[0],coordonnee[1],d1,coordonnee[2])
    csvfile.close()






