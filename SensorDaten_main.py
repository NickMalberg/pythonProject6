import pandas as pd
import seaborn as sns
import math
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans
from typing import List, Tuple
import geopy.distance


def haversine_formel(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin((dlat) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((dlon) / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6_367_000 * c

    return m



path = r'C:####'  # ToDo: use your path

csv_files = glob.glob(os.path.join(path, "*.csv"))

files = []

for name in csv_files:
    out = name.split("\\")[-1]
    out = out.replace(".csv", "")
    files.append(out)

for i in files:
    print(i)

df_names = {}

for f, name in zip(csv_files, files):
    df = pd.read_csv(f)
    df_names[name] = df

names = []
for i in files:
    names.append(f"{i}")


# Rundet alle DFs im Dictionary auf die zweite Nachkommastelle
for i in names:
    df_names[i]["Time"] = df_names[i]["Time (s)"].round(2)

# Um Fehler beim Merge von Gyro und Acc zu vermeiden, müssen diese zusätzlich auf 3 Nachkommastellen gerundet werden
for i in ["Gyroscope", "Accelerometer"]:
    df_names[i]["Time_3"] = df_names[i]["Time (s)"].round(3)


# Vektorisierbare Funktion für die Berechnung der Distanzen zwischen zwei GPS Punkten

def haversine_formel(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin((dlat) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((dlon) / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6_367_000 * c

    return m

# 1. Erstellt zwei Spalten und füllt diese mit den Längen- und Breitengraden der letzten Messung.

df_names["Location"]["Last Lon"] = df_names["Location"]["Longitude (°)"].shift()
df_names["Location"]["Last Lat"] = df_names["Location"]["Latitude (°)"].shift()

# 2. Nutzt Haversine Formel als vektorisierbare Funktion, um die Entfernungen in Metern zu berechnen.


df_names["Location"]["Distance"] = haversine_formel(df_names["Location"]["Longitude (°)"], df_names["Location"]["Latitude (°)"], df_names["Location"]["Last Lon"], df_names["Location"]["Last Lat"])

# Merge die drei relevanten DataFrames

df = df_names["Accelerometer"]
df = df.merge(df_names["Location"], on="Time", how='left')
df = df.merge(df_names["Gyroscope"], how="left", on='Time_3')

#drop irrelevante Columns

df.drop(['Time (s)_y', 'Time_x', 'Time_3', 'Time_y', 'Time (s)'], axis=1, inplace=True)

df = df.rename(columns={"Time (s)_x": "Time (s)"})

#Nur Distance Werte Interpolieren, da diese nicht als Berechnungswerte im ML Algo dienen werden.
#Gyroscop oder andere Werte Linear zu Interpolieren scheint zu fehlern führen zu können.

df["Distance"] = df["Distance"].interpolate()


#Beschleunigungsdaten der Y-Achse durch FFT + Filter glätten
f = df["Acceleration y (m/s^2)"].to_numpy()
t = df["Time (s)"].to_numpy()

n = len(f)
dt = 0.00223

fhat = np.fft.fft(f,n)
PSD = fhat * np.conj(fhat) / n
freq = (1 / (dt*n)) * np.arange(n)
L = np.arange(1,np.floor(n/2), dtype='int')
fig,axs = plt.subplots(2,1)
plt.sca(axs[0])
plt.plot(t,f,color='c', LineWidth=1.5,label="Noisy Data")
plt.xlim(t[0], t[-1])
plt.legend()

plt.sca(axs[1])
plt.plot(freq[L], PSD[L], color='r',LineWidth=1,label='Frequenzen')

plt.legend()
plt.grid()

plt.axis([0,1,0,5000])


plt.show()

indices = PSD > 500
PSDclean = PSD * indices
fhat = indices * fhat
ffilt = np.fft.ifft(fhat)

fig,axs = plt.subplots(2,1)

plt.sca(axs[0])
plt.plot(t,f,color='c', LineWidth=1.5,label="Noisy Data")
plt.xlim(t[0], t[-1])
plt.legend()


plt.sca(axs[1])
plt.plot(t,ffilt,color='k', LineWidth=2,label="Filtered")
plt.legend()
plt.xlim(400,410)

plt.sca(axs[2])
plt.plot(freq[L],PSD[L],color='b', LineWidth=2,label="Noisy")
plt.plot(freq[L],PSDclean[L],color='k',LineWidth=2,label="Filtered")
plt.xlim(0,3)

plt.show()

#Geglättete Daten dem DF hinzufügen
df["cleaned Acc y"] = np.real(ffilt)


# 1. Durchlauf ohne FFT geglättete Daten

df_use = df[['Acceleration y (m/s^2)','Acceleration z (m/s^2)']].dropna(axis=0)
relevant_data = df_use

#K-Means Algorithmus über Sklearn anwenden
kmeans_clustering = KMeans(n_clusters=3, max_iter=50, n_init=50, random_state=0)
kmeans_clustering.fit(relevant_data)


# Labels aus K-Means bekommen und dem DF hinzufügen
df_use["cluster_col"] = kmeans_clustering.labels_


#"Time (s)" wieder dem K-Means Datensatz hinzufügen, um den Datensatz wieder plotten zu können
df_use["Time"] = df["Time (s)"]

df_use[(df_use["Time"] > 200) & (df_use["Time"] < 210) ].plot(x="Time", y=["Acceleration y (m/s^2)", "cluster_col"], figsize=(20,10))

#-> Klassifizierung scheint nicht geklappt zu haben.



# 2. Durchlauf mit FFT geglättete Daten

df_use_2 = df[['cleaned Acc y', "cleaned Acc z"]].dropna(axis=0)

relevant_data = df_use_2


#K-Means Algorithmus über Sklearn anwenden
kmeans_clustering = KMeans(n_clusters=4, max_iter=50, n_init=50, random_state=0)
kmeans_clustering.fit(relevant_data)


# Labels aus K-Means bekommen und dem DF hinzufügen
df_use_2["cluster_col"] = kmeans_clustering.labels_

#Finale Ergebnisse Darstellen

df_use_2["Time"] = df["Time (s)"]
df_use_2["norm_acc"] = -df_use_2["cleaned Acc y"]

df_use_2[(df_use["Time"] > 200) & (df_use["Time"] < 210) ].plot(x="Time", y=['norm_acc', "cluster_col"], figsize=(10,10))