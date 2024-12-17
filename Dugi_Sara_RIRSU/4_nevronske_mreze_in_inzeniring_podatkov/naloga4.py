import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import joblib

output_dir = 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov'
os.makedirs(output_dir, exist_ok=True)
file_path_modded = 'Dugi_Sara_RIRSU/1_regresija/modified_data/modded_data.csv'
df = pd.read_csv(file_path_modded)

# Preverjanje manjkajočih vrednosti in njihovo zapolnjevanje
missing_cols = df.columns[df.isnull().any()]
for col in missing_cols:
    not_null_df = df[df[col].notnull()]
    null_df = df[df[col].isnull()]
    
    X_train = not_null_df.drop(columns=[col])
    y_train = not_null_df[col]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    X_null = null_df.drop(columns=[col])
    df.loc[df[col].isnull(), col] = model.predict(X_null)

# Preverjanje imena ciljne spremenljivke
target_column = 'rented_bike_count'

# Dodajanje treh novih značilnic
df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
df['temp_squared'] = df['temperature'] ** 2
df['humidity_inverse'] = 1 / (df['humidity'] + 1e-3)

# Odprava težav z ničelnimi vrednostmi za log transformacijo
df['temperature'] = df['temperature'].apply(lambda x: x if x > 0 else np.nan)
df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
df['temperature_log'] = np.log1p(df['temperature'])

# Preverjanje in odstranjevanje NaN vrednosti
if df.isnull().values.any():
    print("Podatki vsebujejo manjkajoče vrednosti. Nadomeščam manjkajoče vrednosti...")
    df.fillna(df.median(), inplace=True)

# Standardizacija značilnic
numerical_features = ['temperature', 'humidity', 'temp_humidity_interaction', 'temp_squared', 'humidity_inverse']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Izbira značilnic z metodo "information gain"
X = df.drop(columns=[target_column])
y = df[target_column]

info_gain = mutual_info_regression(X, y)
info_gain_threshold = 0.01
selected_features = X.columns[info_gain > info_gain_threshold]
print("Izbrane značilnice:", selected_features)

# Normalizacija 
X_selected = df[selected_features]

# Shrani seznam vseh značilk uporabljenih med učenjem
joblib.dump(X_selected.columns.tolist(), os.path.join(output_dir, "all_features.pkl"))
print("Seznam vseh značilk je shranjen.")


# Regresijski algoritmi
regression_algorithms = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=1234),
    "GradientBoosting": GradientBoostingRegressor(random_state=1234),
    "AdaBoost": AdaBoostRegressor(random_state=1234),
    "MLPRegressor": MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=1234)
}

# Shranjevanje metrik za regresijo
regression_results = {algo: {'MAE': [], 'MSE': [], 'ExplainedVariance': []} for algo in regression_algorithms.keys()}

kf = KFold(n_splits=5, shuffle=True, random_state=1234)

for algo_name, model in regression_algorithms.items():
    for train_index, test_index in kf.split(X_selected):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Treniranje modela
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Izračun metrik
        regression_results[algo_name]['MAE'].append(mean_absolute_error(y_test, y_pred))
        regression_results[algo_name]['MSE'].append(mean_squared_error(y_test, y_pred))
        regression_results[algo_name]['ExplainedVariance'].append(explained_variance_score(y_test, y_pred))

# Povprečne vrednosti za regresijo
average_regression_results = {algo: {metric: np.mean(values) for metric, values in metrics.items()} for algo, metrics in regression_results.items()}
print("\nPovprečni rezultati za regresijo:")
for algo, metrics in average_regression_results.items():
    print(f"\n### {algo} ###")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Grafikoni za regresijo
regression_metrics = ['MAE', 'MSE', 'ExplainedVariance']

for metric in regression_metrics:
    plt.figure(figsize=(10, 6))
    data = [regression_results[algo][metric] for algo in regression_algorithms.keys()]
    plt.boxplot(data, labels=regression_algorithms.keys(), showmeans=True)
    plt.title(f'Regresija - Boxplot za {metric}')
    plt.ylabel(metric)
    plt.xlabel('Algoritem')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, f'regression_{metric}_boxplot.png'))
    plt.show()

for metric in regression_metrics:
    plt.figure(figsize=(10, 6))
    values = [average_regression_results[algo][metric] for algo in regression_algorithms.keys()]
    plt.bar(regression_algorithms.keys(), values)
    plt.title(f'Regresija - Stolpični diagram za {metric}')
    plt.ylabel(metric)
    plt.xlabel('Algoritem')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, f'regression_{metric}_barplot.png'))
    plt.show()

# Klasifikacijski problem
classification_algorithms = {
    "RandomForestClassifier": RandomForestClassifier(random_state=1234),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=1234),
    "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME', random_state=1234),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=1234)
}

classification_results = {algo: {'Accuracy': [], 'F1': [], 'Precision': [], 'Recall': []} for algo in classification_algorithms.keys()}

for algo_name, model in classification_algorithms.items():
    for train_index, test_index in kf.split(X_selected):
        X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
        y_train, y_test = (y > y.mean()).astype(int).iloc[train_index], (y > y.mean()).astype(int).iloc[test_index]

        # Treniranje modela
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Izračun metrik
        classification_results[algo_name]['Accuracy'].append(accuracy_score(y_test, y_pred))
        classification_results[algo_name]['F1'].append(f1_score(y_test, y_pred, average='weighted'))
        classification_results[algo_name]['Precision'].append(precision_score(y_test, y_pred, average='weighted'))
        classification_results[algo_name]['Recall'].append(recall_score(y_test, y_pred, average='weighted'))

# Povprečne vrednosti za klasifikacijo
average_classification_results = {algo: {metric: np.mean(values) for metric, values in metrics.items()} for algo, metrics in classification_results.items()}
print("\nPovprečni rezultati za klasifikacijo:")
for algo, metrics in average_classification_results.items():
    print(f"\n### {algo} ###")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        

import joblib

scalers_dir = 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/scalers'
os.makedirs(scalers_dir, exist_ok=True)

adaboost_model = AdaBoostRegressor(random_state=1234)
adaboost_model.fit(X_selected, y)
joblib.dump(adaboost_model, os.path.join(scalers_dir, 'AdaBoost_Regresija.joblib'))
print("AdaBoost model shranjen.")

scaler_standard = StandardScaler()
df[numerical_features] = scaler_standard.fit_transform(df[numerical_features])
joblib.dump(scaler_standard, os.path.join(scalers_dir, 'scaler_standard_new.pkl'))
print("StandardScaler shranjen.")