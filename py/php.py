# ----------- Bibliothèque ------------

import csv
import sys
import os

# ---------------------- PHP ---------------------

if __name__ == "__main__":
    coord = [sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]]
    try:
        os.remove(r'./csv/coordonnee.csv')
        with open(r'./csv/coordonnee.csv', mode='w', encoding='utf-8') as a:
            witter = csv.writer(a)
            witter.writerow(coord)
    except FileNotFoundError:
        with open(r'./csv/coordonnee.csv', mode='w', encoding='utf-8') as a:
            witter = csv.writer(a)
            witter.writerow(coord)
