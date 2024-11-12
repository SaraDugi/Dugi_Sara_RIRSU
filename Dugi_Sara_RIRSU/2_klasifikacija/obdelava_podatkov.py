import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

# Glavni del naloge
# 1. Obdelava podatkov
data_dir = r"Dugi_Sara_RIRSU\2_klasifikacija\shapes"
save_dir = r"Dugi_Sara_RIRSU\2_klasifikacija" 
categories = ['circles', 'squares', 'triangles']
image_size = (64, 64) 

image_paths = []
labels = []

for idx, category in enumerate(categories):
    folder = os.path.join(data_dir, category)
    for filename in os.listdir(folder):
        if filename.endswith('.png'):
            image_paths.append(os.path.join(folder, filename))
            labels.append(idx)  # 0 = krogi, 1 = kvadrati, 2 = trikotniki

# Funkcija za nalaganje in pretvorbo slike v enodimenzionalno polje
def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')  
        img = img.resize(image_size)  
        img_array = np.array(img).flatten() 
    return img_array

# pretvorba v polje
X = np.array([load_image(path) for path in image_paths])
y = np.array(labels)

# Izris ene slike iz vsake kategorije
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for i, category in enumerate(categories):
    sample_image = Image.open(image_paths[labels.index(i)])
    axs[i].imshow(sample_image.convert('L'), cmap='gray')
    axs[i].set_title(category)
    axs[i].axis('off')
plt.savefig(os.path.join(save_dir, 'sample_images.png'))
plt.show()

# 2. Izgradnja napovednega modela (odločitveno drevo)
# Razdelitev podatkov na učno in testno množico
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4321, shuffle=True)

# Gradnja modela - odločitveno drevo
clf_tree = DecisionTreeClassifier(random_state=1234)
clf_tree.fit(X_train, y_train)

plt.figure(figsize=(20, 10))
plot_tree(clf_tree, filled=True, feature_names=None, class_names=categories)
plt.savefig(os.path.join(save_dir, 'decision_tree.png'))
plt.show()

# 3. Ovrednotenje napovednega modela
# Napovedovanje vrednosti
y_pred_tree = clf_tree.predict(X_test)

# Izračun meritev
accuracy_tree = accuracy_score(y_test, y_pred_tree)
f1_tree = f1_score(y_test, y_pred_tree, average='weighted')
precision_tree = precision_score(y_test, y_pred_tree, average='weighted')
recall_tree = recall_score(y_test, y_pred_tree, average='weighted')

print(f"\nTočnost (Decision Tree): {accuracy_tree:.4f}")
print(f"Utežena F1 vrednost (Decision Tree): {f1_tree:.4f}")
print(f"Utežena preciznost (Decision Tree): {precision_tree:.4f}")
print(f"Utežen priklic (Decision Tree): {recall_tree:.4f}\n")


# Dodatni del naloge

# 1. Gradnja napovednega modela (k-najbližji sosedje)
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(X_train, y_train)

# Napovedovanje vrednosti
y_pred_knn = clf_knn.predict(X_test)

# Izračun meritev
accuracy_knn = accuracy_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
recall_knn = recall_score(y_test, y_pred_knn, average='weighted')

print(f"\nTočnost (k-NN): {accuracy_knn:.4f}")
print(f"Utežena F1 vrednost (k-NN): {f1_knn:.4f}")
print(f"Utežena preciznost (k-NN): {precision_knn:.4f}")
print(f"Utežen priklic (k-NN): {recall_knn:.4f}\n")

# 2. Primerjava rezultatov obeh algoritmov v grafu
metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
tree_scores = [accuracy_tree, f1_tree, precision_tree, recall_tree]
knn_scores = [accuracy_knn, f1_knn, precision_knn, recall_knn]

x = np.arange(len(metrics))

plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, tree_scores, 0.4, label='Decision Tree')
plt.bar(x + 0.2, knn_scores, 0.4, label='k-NN')

plt.xticks(x, metrics)
plt.ylabel('Rezultat')
plt.title('Primerjava odločitvenega drevesa in k-NN')
plt.legend()
plt.savefig(os.path.join(save_dir, 'primerjava.png'))
plt.show()