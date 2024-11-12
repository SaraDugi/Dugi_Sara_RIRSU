import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

file_path_modded = 'Dugi_Sara_RIRSU/1_regresija/modified_data/modded_data.csv'
modded_df = pd.read_csv(file_path_modded)

# Razdelitev podatkov na učno in testno množico
# tako, da bo testna množica zajemala 30% 
# vseh podatkov pri čemer naj bo parameter random_state nastavljen na vrednost 1234.
x = modded_df.drop(columns=['rented_bike_count'])
y = modded_df['rented_bike_count']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.30, random_state = 1234)

#Gradnjo (učenje) napovednega modela implementirajte z uporabo algoritma linearne regresije nad podatki iz učne množice.
model = LinearRegression()
model.fit(x_train, y_train)

#Pridobivanje napovedane vrednosti z uporabo zgrajenega napovednega modela za vsak primerek iz množice testnih podatkov.
y_pred = model.predict(x_test)

print("\nNapovedane vrednosti:\n ", y_pred)
print("\nDejanske vrednosti:\n ", y_test.values)

# Izračun sledečih metrik nad testnimi podatki:
# povprečna absolutna napaka,
mae = mean_absolute_error(y_test, y_pred)
print(f'\nPovprecna absolutna napaka (MAE):\n {mae}')

#povprečna kvadratna napaka,
mse = mean_squared_error(y_test, y_pred)
print(f'\nPovprecna kvadratna napaka (MSE):\n {mse}')

#vrednost razložene variance.
exolained_var = explained_variance_score(y_test, y_pred)
print(f'\nVrednost razložene variance: {exolained_var}')