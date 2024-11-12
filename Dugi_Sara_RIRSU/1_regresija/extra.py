import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

file_path_modded = 'Dugi_Sara_RIRSU/1_regresija/modified_data/modded_data.csv'
modded_df = pd.read_csv(file_path_modded)

X = modded_df.drop(columns=['rented_bike_count'])
y = modded_df['rented_bike_count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1234)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
explained_var_linear = explained_variance_score(y_test, y_pred_linear)

print("### Linearna regresija ###")
print(f"Povprečna absolutna napaka (MAE): {mae_linear}")
print(f"Povprečna kvadratna napaka (MSE): {mse_linear}")
print(f"Vrednost razložene variance: {explained_var_linear}")

# RIDGE
ridge_model = Ridge(alpha = 1)
ridge_model.fit(X_train, y_train)

y_pred_ridge = ridge_model.predict(X_test)

mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
explained_var_ridge = explained_variance_score(y_test, y_pred_ridge)

print("\n### Ridge regresija ###")
print(f"Povprečna absolutna napaka (MAE): {mae_ridge}")
print(f"Povprečna kvadratna napaka (MSE): {mse_ridge}")
print(f"Vrednost razložene variance: {explained_var_ridge}")

plt.figure(figsize = (12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha = 0.6, color = "blue" , label='Napovedi Linearne regresije')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', label='Idealna napoved')
plt.xlabel('Dejanske vrednosti')
plt.ylabel('Napovedane vrednosti')
plt.title('Linearna regresija: Dejanske vs. Napovedane')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, alpha=0.6, color='green', label='Napovedi Ridge regresije')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', label='Idealna napoved')
plt.xlabel('Dejanske vrednosti')
plt.ylabel('Napovedane vrednosti')
plt.title('Ridge regresija: Dejanske vs. Napovedane')
plt.legend()

plt.tight_layout()
plt.show()