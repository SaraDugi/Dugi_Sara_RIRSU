import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBRegressor, XGBClassifier 

output_dir = os.path.dirname(__file__)

file_path = 'Dugi_Sara_RIRSU/1_regresija/original_data/bike_data.csv'
df = pd.read_csv(file_path)

df.fillna(df.mean(numeric_only=True), inplace=True)
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df.drop('date', axis=1, inplace=True)
df['seasons'] = df['seasons'].astype('category').cat.codes
df['holiday'] = df['holiday'].astype('category').cat.codes
df['work_hours'] = df['work_hours'].astype('category').cat.codes

X_reg = df.drop(columns=['rented_bike_count'])
y_reg = df['rented_bike_count']

data_dir = 'Dugi_Sara_RIRSU/2_klasifikacija/shapes'
categories = ['circles', 'squares', 'triangles']
image_size = (64, 64)
image_paths = []
labels = []

for idx, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            image_paths.append(os.path.join(folder, filename))
            labels.append(idx)

def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')
        img = img.resize(image_size)
        return np.array(img).flatten()

X_clf = np.array([load_image(path) for path in image_paths])
y_clf = np.array(labels)

# Regresorji in klasifikatorji
regressors = {
    'Bagging': BaggingRegressor(random_state=1234),
    'RandomForest': RandomForestRegressor(random_state=1234),
    'AdaBoost': AdaBoostRegressor(random_state=1234),
    'XGBoost': XGBRegressor(random_state=1234)
}

classifiers = {
    'Bagging': BaggingClassifier(random_state=1234),
    'RandomForest': RandomForestClassifier(random_state=1234),
    'AdaBoost': AdaBoostClassifier(random_state=1234),
    'XGBoost': XGBClassifier(random_state=1234)
}

kf = KFold(n_splits=5, shuffle=True, random_state=1234)

# regresijska evolucija?
regression_results = []
regression_results_detailed = {name: {'MAE': [], 'MSE': [], 'ExplainedVariance': []} for name in regressors}

#for zanka namesto cross_validate
for name, model in regressors.items():
    cv_results = cross_validate(model, X_reg, y_reg, cv=kf,
                                scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'explained_variance'])
    regression_results.append({
        'Model': name,
        'MAE': -cv_results['test_neg_mean_absolute_error'].mean(),
        'MSE': -cv_results['test_neg_mean_squared_error'].mean(),
        'ExplainedVariance': cv_results['test_explained_variance'].mean()
    })
    regression_results_detailed[name]['MAE'].extend(-cv_results['test_neg_mean_absolute_error'])
    regression_results_detailed[name]['MSE'].extend(-cv_results['test_neg_mean_squared_error'])
    regression_results_detailed[name]['ExplainedVariance'].extend(cv_results['test_explained_variance'])

regression_df = pd.DataFrame(regression_results)
print("\n", regression_df, "\n")

# klasifikacijska evolucija? 
classification_results = []
classification_results_detailed = {name: {'Accuracy': [], 'F1': [], 'Precision': [], 'Recall': []} for name in classifiers}

for name, model in classifiers.items():
    cv_results = cross_validate(model, X_clf, y_clf, cv=kf, scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'])
    classification_results.append({
        'Model': name,
        'Accuracy': cv_results['test_accuracy'].mean(),
        'F1': cv_results['test_f1_weighted'].mean(),
        'Precision': cv_results['test_precision_weighted'].mean(),
        'Recall': cv_results['test_recall_weighted'].mean()
    })
    classification_results_detailed[name]['Accuracy'].extend(cv_results['test_accuracy'])
    classification_results_detailed[name]['F1'].extend(cv_results['test_f1_weighted'])
    classification_results_detailed[name]['Precision'].extend(cv_results['test_precision_weighted'])
    classification_results_detailed[name]['Recall'].extend(cv_results['test_recall_weighted'])

classification_df = pd.DataFrame(classification_results)
print("\n", classification_df, "\n")

for metric in ['MAE', 'MSE', 'ExplainedVariance']:
    plt.figure()
    plt.bar(regression_df['Model'], regression_df[metric])
    plt.title(f'Regression {metric} by Model')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.savefig(os.path.join(output_dir, f'Regression_{metric}_by_Model.png'))
    plt.close()

for metric in ['Accuracy', 'F1', 'Precision', 'Recall']:
    plt.figure()
    plt.bar(classification_df['Model'], classification_df[metric])
    plt.title(f'Classification {metric} by Model')
    plt.xlabel('Model')
    plt.ylabel(metric)
    plt.savefig(os.path.join(output_dir, f'Classification_{metric}_by_Model.png'))
    plt.close()

for metric in ['MAE', 'MSE', 'ExplainedVariance']:
    plt.figure()
    data = [regression_results_detailed[name][metric] for name in regressors]
    plt.boxplot(data, labels=regressors.keys())
    plt.title(f'Regression {metric} Boxplot')
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.savefig(os.path.join(output_dir, f"Regression_{metric}_Boxplot.png"))
    plt.close()

for metric in ['Accuracy', 'F1', 'Precision', 'Recall']:
    plt.figure()
    data = [classification_results_detailed[name][metric] for name in classifiers]
    plt.boxplot(data, labels=classifiers.keys())
    plt.title(f'Classification {metric} Boxplot')
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.savefig(os.path.join(output_dir, f"Classification_{metric}_Boxplot.png"))
    plt.close()