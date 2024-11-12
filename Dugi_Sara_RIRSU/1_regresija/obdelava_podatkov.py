import os
import pandas as pd
import matplotlib.pyplot as mt 

file_path = 'Dugi_Sara_RIRSU/1_regresija/original_data/bike_data.csv'
df = pd.read_csv(file_path)

# manjkajoci podatki
missing_values = df.isnull().sum()
print("Število vrstic, kjer so manjkajoči podatki, za vsak stolpec:")
print(missing_values)

# polnjenje z povprecnimi vrednosti
df.fillna(df.mean(numeric_only=True), inplace=True)

# Convert datum v 3
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df.drop('date', axis=1, inplace=True)

# Convert v num
df['seasons'] = df['seasons'].astype('category').cat.codes
df['holiday'] = df['holiday'].astype('category').cat.codes
df['work_hours'] = df['work_hours'].astype('category').cat.codes

# opis
description = df.describe()
print(description)

mt.figure(figsize=(10, 6))
mt.hist(df['temperature'], bins=20, color='skyblue', edgecolor="black")
mt.title("Histogram za temperaturo")
mt.xlabel("Temperatura (Celzije)")
mt.ylabel("Frekvenčnost")
mt.grid(axis='y', alpha=0.75)
mt.show()

mt.figure(figsize=(10, 6))
mt.scatter(df['temperature'], df['rented_bike_count'], color="orange", alpha=0.6)
mt.title("Graf razstrosa")
mt.xlabel("Temperatura (Celzije)")
mt.ylabel("Rented Bike Count")
mt.grid(True)
mt.show()