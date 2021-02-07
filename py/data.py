# ----------- Bibliothèque ------------

import csv
from itertools import islice

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

# lignes_metro = {1: ["Chatelet-Les-Halles", "Louvre-Rivoli", "Palais-Royal", "Tuileries", "Concorde"],
#                 4: ["Chatelet-Les-Halles", "Cite", "Odeon"],
#                 8: ["Concorde", "Varenne", "Invalide"],
#                 10: ["Duroc", "Vaneau", "Sevres", "Mabillon", "Odeon"],
#                 12: ["Concorde", "Assemblee Nationale", "Solferino-Bellechase", "Rue du bac", "Sevres"],
#                 13: ["Duroc", "Saint-François-Xavier", "Varenne", "Invalide"]}

lignes_index = {1: [6, 7, 8, 9, 10],
                4: [6, 5, 4],
                8: [10, 15, 14], 10: [0, 1, 2, 3, 4],
                12: [10, 11, 12, 13, 2],
                13: [0, 16, 14, 15]}

# ----------- Donées (CSV + Autres) --------------

csvfile = open('./csv/station.csv', 'r')
readCSV = csv.reader(csvfile)

station_csv = []
for data in islice(readCSV, 1, None):
    station_csv.append([data[0], float(data[2]), float(data[3])])

long = [p[1] for p in station_csv]
lat = [p[2] for p in station_csv]
station = [s[0] for s in station_csv]