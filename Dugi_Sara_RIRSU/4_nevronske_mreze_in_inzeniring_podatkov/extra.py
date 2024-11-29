import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from skimage import filters
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

output_dir = 'Dugi_Sara_RIRSU/4_nevronske_mreže_in_inženiring_podatkov'
os.makedirs(output_dir, exist_ok=True)
image_data_dir = 'Dugi_Sara_RIRSU/2_klasifikacija/shapes'
categories = ['circles', 'squares', 'triangles']
image_size = (64, 64)

# Procesiranje slik za klasifikacijo
images = []
labels = []

for idx, category in enumerate(categories):
    category_path = os.path.join(image_data_dir, category)
    for file_name in os.listdir(category_path):
        image_path = os.path.join(category_path, file_name)
        image = imread(image_path)
        image = rgb2gray(resize(image, image_size))  # Pretvorba v sivinsko sliko in sprememba velikosti
        
        # detekcija robov
        edges = filters.sobel(image)
        
        # Normalizacija in pretvorba v eno samo vrsto
        normalized_image = edges / np.max(edges)
        images.append(normalized_image.flatten())
        labels.append(idx)

X_images = np.array(images)
y_images = np.array(labels)

print("Procesiranje slik zaključeno.")
print(f"Število slik: {len(X_images)}, Oblika: {X_images.shape}")

# 5-kratno navzkrižno preverjanje
kf = KFold(n_splits=5, shuffle=True, random_state=1234)

# Modeli za klasifikacijo
classification_algorithms = {
    "RandomForestClassifier": RandomForestClassifier(random_state=1234),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=1234),
    "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME', random_state=1234),
    "MLPClassifier_Default": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=1234),
    "MLPClassifier_Custom": MLPClassifier(hidden_layer_sizes=(128, 128, 64), activation='relu', solver='adam', learning_rate_init=0.001, max_iter=1000, random_state=1234)
}

classification_results = {algo: {'Točnost': [], 'F1': [], 'Preciznost': [], 'Priklic': []} for algo in classification_algorithms.keys()}

# Učenje modelov za klasifikacijo
for algo_name, model in classification_algorithms.items():
    for train_index, test_index in kf.split(X_images):
        X_train, X_test = X_images[train_index], X_images[test_index]
        y_train, y_test = y_images[train_index], y_images[test_index]

        # Učenje in napovedovanje
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Merila uspešnosti
        classification_results[algo_name]['Točnost'].append(accuracy_score(y_test, y_pred))
        classification_results[algo_name]['F1'].append(f1_score(y_test, y_pred, average='weighted'))
        classification_results[algo_name]['Preciznost'].append(precision_score(y_test, y_pred, average='weighted'))
        classification_results[algo_name]['Priklic'].append(recall_score(y_test, y_pred, average='weighted'))

# Povprečni rezultati za klasifikacijo
average_classification_results = {algo: {metric: np.mean(values) for metric, values in metrics.items()} for algo, metrics in classification_results.items()}

# Izpis rezultatov za vse modele
print("\nPovprečni rezultati za klasifikacijo:")
for algo, metrics in average_classification_results.items():
    print(f"\n### {algo} ###")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Originalni podatki (brez predprocesiranja)
classification_algorithms_no_preprocessing = {
    "RandomForestClassifier": RandomForestClassifier(random_state=1234),
    "GradientBoostingClassifier": GradientBoostingClassifier(random_state=1234),
    "AdaBoostClassifier": AdaBoostClassifier(algorithm='SAMME', random_state=1234),
}

results_no_preprocessing = {algo: {'Točnost': [], 'F1': [], 'Preciznost': [], 'Priklic': []} for algo in classification_algorithms_no_preprocessing.keys()}

for algo_name, model in classification_algorithms_no_preprocessing.items():
    for train_index, test_index in kf.split(X_images):  # Popravljeno, da uporablja X_images in y_images
        X_train, X_test = X_images[train_index], X_images[test_index]
        y_train, y_test = y_images[train_index], y_images[test_index]

        # Učenje in napovedovanje
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Merila uspešnosti
        results_no_preprocessing[algo_name]['Točnost'].append(accuracy_score(y_test, y_pred))
        results_no_preprocessing[algo_name]['F1'].append(f1_score(y_test, y_pred, average='weighted'))
        results_no_preprocessing[algo_name]['Preciznost'].append(precision_score(y_test, y_pred, average='weighted'))
        results_no_preprocessing[algo_name]['Priklic'].append(recall_score(y_test, y_pred, average='weighted'))

# Povprečni rezultati brez predprocesiranja
average_results_no_preprocessing = {algo: {metric: np.mean(values) for metric, values in metrics.items()} for algo, metrics in results_no_preprocessing.items()}
print("\nPovprečni rezultati brez predprocesiranja:")
for algo, metrics in average_results_no_preprocessing.items():
    print(f"\n### {algo} ###")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")