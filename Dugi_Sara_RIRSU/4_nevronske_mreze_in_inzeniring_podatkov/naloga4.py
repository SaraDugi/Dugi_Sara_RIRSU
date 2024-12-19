import pandas as pd
import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression, Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, explained_variance_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.neural_network import MLPRegressor

df_bike_data = pd.read_csv('Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/bike_data.csv')

# Sprememba date v tri stolpce
df_bike_data[['day', 'month', 'year']] = df_bike_data['date'].str.split('/', expand=True)
df_bike_data['day'] = pd.to_numeric(df_bike_data['day'])
df_bike_data['month'] = pd.to_numeric(df_bike_data['month'])
df_bike_data['year'] = pd.to_numeric(df_bike_data['year'])
df_bike_data = df_bike_data.drop('date', axis=1)

# Sprememba kategoričnih podatkov
categoric_columns = ['seasons', 'holiday', 'work_hours']
df_bike_data = pd.get_dummies(df_bike_data, columns=categoric_columns)

#Uporaba regresije za manjkajoče podatke 
train = df_bike_data[df_bike_data['dew_point_temperature'].notnull()]
test = df_bike_data[df_bike_data['dew_point_temperature'].isnull()]

X_train = train.drop('dew_point_temperature', axis=1)
y_train = train['dew_point_temperature']
X_test = test.drop('dew_point_temperature', axis=1)

#Model linearne regresije 
model = LinearRegression()
model.fit(X_train, y_train)

#Napovedane vrednosti
predicted_values = model.predict(X_test)

#Zamenjava vrednosti z napovedanimi
df_bike_data.loc[df_bike_data['dew_point_temperature'].isnull(), 'dew_point_temperature'] = predicted_values

df_bike_data.isnull().sum()

df_bike_data

output_file_path = 'bike_data_podatkiRegresija.csv' 
df_bike_data.to_csv(output_file_path, index=False)

def categorize_hour(hour):
    if 5 <= hour <= 11: 
        return 0.7 
    elif 12 <= hour <= 16:  
        return 0.8  
    elif 17 <= hour <= 20: 
        return 0.6 
    else:  
        return 0.1  
    
df_bike_data['is_raining'] = df_bike_data['rainfall'].apply(lambda x: 1 if x > 0 else 0)

df_bike_data['feels_like_temperature'] = (
    df_bike_data['temperature'] - ((0.55 - 0.0055 * df_bike_data['humidity']) * 
    (df_bike_data['temperature'] - 14.5))
)

df_bike_data['hour_category'] = df_bike_data['hour'].apply(categorize_hour)

df_bike_data

df_bike_data.hist(bins=15, figsize=(15, 10))

output_file_path = 'bike_data_dodaneZnacilnice.csv' 
df_bike_data.to_csv(output_file_path, index=False)

df_bike_data = pd.read_csv('bike_data_dodaneZnacilnice.csv')

# df_bike_data['rented_bike_count'] = np.log1p(df_bike_data['rented_bike_count'])
df_bike_data['solar_radiation'] = np.log1p(df_bike_data['solar_radiation'])
df_bike_data['wind_speed'] = np.log1p(df_bike_data['wind_speed'])

df_bike_data['rainfall'] = np.log1p(df_bike_data['rainfall'])
df_bike_data['snowfall'] = np.log1p(df_bike_data['snowfall'])

df_bike_data.hist(bins=15, figsize=(15, 10))

scaler_standard = StandardScaler()
scaler_minmax = MinMaxScaler()

scaler_standard.fit(df_bike_data[['temperature', 'humidity', 'wind_speed', 'feels_like_temperature', 'dew_point_temperature']])
scaler_minmax.fit(df_bike_data[['solar_radiation']])

joblib.dump(scaler_standard, 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/models/scaler_standard_new.pkl')
joblib.dump(scaler_minmax, 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/scaler_minmax_new.pkl')

df_bike_data.hist(bins=15, figsize=(15, 10))


X = df_bike_data.drop('rented_bike_count', axis=1)  
y = df_bike_data['rented_bike_count']  

mi_scores = mutual_info_regression(X, y)

mi_series = pd.Series(mi_scores, index=X.columns)
mi_series = mi_series.sort_values(ascending=False)

print(mi_series)

# selected_features = mi_series[mi_series > 0.05].index

selected_features = [
    'temperature', 'feels_like_temperature', 'hour', 'month',
    'dew_point_temperature', 'seasons_Winter', 'solar_radiation',
    'work_hours_No', 'work_hours_Yes', 'hour_category', 'humidity',
    'rainfall', 'visibility', 'seasons_Summer', 'seasons_Autumn', 'is_raining'
]


print(f"Selected features based on mutual information (threshold 0.05):")
print("Selected features based on mutual information:", selected_features)

X_reg_bike = X[selected_features]
y_reg_bike = df_bike_data['rented_bike_count']  

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)


regression_models = {
    'Bagging Regresija': BaggingRegressor,
    'Random Forest Regresija': RandomForestRegressor,
    'AdaBoost Regresija': AdaBoostRegressor,
    'XGBoost Regresija': XGBRegressor,
    'Neural Network Regresija': MLPRegressor
}


regression_results = {model_name: {'MSE': [], 'MAE': [], 'EVS': []} for model_name in regression_models}


for model_name, model in regression_models.items():
    for train_index, test_index in kfold.split(X_reg_bike):
        X_train, X_test = X_reg_bike.iloc[train_index], X_reg_bike.iloc[test_index]
        y_train, y_test = y_reg_bike.iloc[train_index], y_reg_bike.iloc[test_index]

       
        t = model(random_state=1234)

        t.fit(X_train, y_train)

        y_pred = t.predict(X_test)

        regression_results[model_name]['MSE'].append(mean_squared_error(y_test, y_pred))
        regression_results[model_name]['MAE'].append(mean_absolute_error(y_test, y_pred))
        regression_results[model_name]['EVS'].append(explained_variance_score(y_test, y_pred))

regression_avg_metrics = {}

for model_name, metrics in regression_results.items():
    averaged_metrics = {}
    for metric, values in metrics.items():
        averaged_metrics[metric] = np.mean(values)
    regression_avg_metrics[model_name] = averaged_metrics

print("Regression Results:")
for model_name, metrics in regression_avg_metrics.items():
    print(f"{model_name} - Average MSE: {metrics['MSE']:.4f}, Average MAE: {metrics['MAE']:.4f}, Average EVS: {metrics['EVS']:.4f}")

for x in regression_results:
    print(x)
    for y in regression_results[x]:
        print(y, regression_results[x][y])
        
print("\nSaving models...")
for model_name, model in regression_models.items():
    model_instance = model(random_state=1234)
    model_instance.fit(X_reg_bike, y_reg_bike)  # Train on full dataset
    model_path = f"Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/models_new/{model_name.replace(' ', '_')}.joblib"
    joblib.dump(model_instance, model_path)
    print(f"{model_name} saved to {model_path}")


joblib.dump(scaler_standard, 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/models/scaler_standard.pkl')
joblib.dump(scaler_minmax, 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/models/scaler_minmax.pkl')
metrics_list = ['MSE', 'MAE', "EVS"]

for metric in metrics_list:

    metric_values = [regression_avg_metrics[model][metric] for model in regression_models.keys()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(regression_models.keys(), metric_values, color='skyblue')
    plt.title(f"Regression {metric} Comparison")
    plt.xlabel("Model")
    plt.ylabel(metric)

    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:.2f}", ha='center', va='bottom')

    plt.show()

for metric in metrics_list:
    plt.figure(figsize=(8, 5))
    plt.boxplot([regression_results[model][metric] for model in regression_models.keys()], labels=regression_models.keys())
    plt.title(f"Regression {metric} Distribution Across Folds")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.show()

pot_do_map = 'Dugi_Sara_RIRSU/4_nevronske_mreze_in_inzeniring_podatkov/shapes'
podatki_slike = []

for kategorija in os.listdir(pot_do_map):
    folder_path = os.path.join(pot_do_map, kategorija)
    for imeSlike in os.listdir(folder_path):
        potDoSlike = os.path.join(folder_path, imeSlike)
        try:
            image = Image.open(potDoSlike).convert('L')
            #display(image)
            image_array = np.array(image)
            
            #Normalizacija
            image_array = image_array / 255.0
            
            polje_slike = image_array.flatten()
            podatki_slike.append([polje_slike, kategorija])
        except Exception as e:
            print(f"Napaka pri procesiranju: {potDoSlike}: {e}")
            
X_class_shapes, y_class_shapes = zip(*podatki_slike)
X_train_shapes, X_test_shapes, y_train_shapes, y_test_shapes = train_test_split(X_class_shapes, y_class_shapes, test_size=0.2, random_state=4321, shuffle=True)

# model_DTC = DecisionTreeClassifier(random_state=1234)
# model_DTC.fit(X_train_shapes, y_train_shapes)
# 
# y_pred_shapes = model_DTC.predict(X_test_shapes)
# accuracy_shapes = accuracy_score(y_test_shapes, y_pred_shapes)
# print(f"(Točnost): {accuracy_shapes:.4f}")
# 
# precision_shapes, recall_shapes, f1_shapes, _ = precision_recall_fscore_support(y_test_shapes, y_pred_shapes, average='weighted')
# print(f"(Utežena preciznost): {precision_shapes:.4f}")
# print(f"(Utežen priklic): {recall_shapes:.4f}")
# print(f"(Utežena F1 vrednost): {f1_shapes:.4f}")

from sklearn.neural_network import MLPClassifier

kfold = KFold(n_splits=5, shuffle=True, random_state=1234)

classification_models = {
    "Bagging Classifier": BaggingClassifier,
    "Random Forest Classifier": RandomForestClassifier,
    "AdaBoost Classifier": AdaBoostClassifier,
    "XGBoost Classifier": XGBClassifier,
    'NeuralNetwork': MLPClassifier
}

classification_results = {model_name: {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': []} for model_name in classification_models}

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y_class_shapes)

for model_name, model in classification_models.items():
    for train_index, test_index in kfold.split(X_class_shapes, y_encoded):
        
        t = model(random_state=1234)
        
        X_train, X_test = np.array(X_class_shapes)[train_index], np.array(X_class_shapes)[test_index]
        y_train, y_test = np.array(y_encoded)[train_index], np.array(y_encoded)[test_index]
            
        t.fit(X_train, y_train)
        y_pred = t.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

        classification_results[model_name]['Accuracy'].append(accuracy)
        classification_results[model_name]['Precision'].append(precision)
        classification_results[model_name]['Recall'].append(recall)
        classification_results[model_name]['F1 Score'].append(f1)

classification_avg_metrics = {
    model_name: {metric: np.mean(values) for metric, values in metrics.items()}
    for model_name, metrics in classification_results.items()
}

print("Classification Results:")
for model_name, metrics in classification_avg_metrics.items():
    print(f"{model_name} - Average Accuracy: {metrics['Accuracy']:.4f}, Average Precision: {metrics['Precision']:.4f}, Average Recall: {metrics['Recall']:.4f}, Average F1 Score: {metrics['F1 Score']:.4f}")


for x in classification_results:
  print(x)
  for y in classification_results[x]:
    print(y, classification_results[x][y])


models = list(classification_models.keys())  # List of models without DTC

# You don't need the DTC metrics anymore, so just skip that part
# 'dtc_metrics' is no longer required

for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    # Collect the metrics for the models without DTC
    metric_values = [classification_avg_metrics[model][metric] for model in models]

    # No need for DTC value, just use the other models
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, metric_values, color='skyblue')
    plt.title(f"Classification {metric} Comparison")
    plt.xlabel("Model")
    plt.ylabel(metric)

    for bar, value in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:.4f}", ha='center', va='bottom')

    plt.show()

for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    plt.figure(figsize=(10, 6))
    # Collect the boxplot data for models without DTC
    box_data = [classification_results[model][metric] for model in models]
    plt.boxplot(box_data, labels=models)
    plt.title(f"Classification {metric} Distribution Across Folds")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.show()
