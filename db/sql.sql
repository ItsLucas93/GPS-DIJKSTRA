-- Il n'est pas nécessaire d'utiliser les bases de données car le programme n'exploite pas cette partie (en CSV ou JSON)
-- Il pourrait être activé en option dans une future update

-- <!------- LINEAR COMMAND -------!>

CREATE DATABASE IF NOT EXISTS MBDD_Lucas;
USE MBDD_Lucas;

CREATE TABLE IF NOT EXISTS eleve(id_eleve INT PRIMARY KEY AUTO_INCREMENT, prenom varchar(255) NOT NULL, longitude FLOAT NOT NULL, latitude FLOAT NOT NULL);
CREATE TABLE IF NOT EXISTS station(NOM_station varchar(255) PRIMARY KEY, longitude FLOAT NOT NULL, latitude FLOAT NOT NULL);
CREATE TABLE IF NOT EXISTS groupe(NOM varchar(255) PRIMARY KEY, id_eleve INT, NOM_station varchar(255), distance FLOAT, FOREIGN KEY (id_eleve) REFERENCES eleve(id_eleve), FOREIGN KEY (NOM_station) REFERENCES station(NOM_station));
CREATE TABLE IF NOT EXISTS trajet(path TEXT);

-- distance : 1000 * 6372.954775981000 * ACOS(SIN(eleve.longitude * math.pi / 180) * SIN(station.longitude * math.pi / 180)) + COS(eleve.longitude * math.pi / 180) * COS(station.longitude * math.pi / 180) * COS(* math.pi / 180(eleve.latitude - station.longitude))