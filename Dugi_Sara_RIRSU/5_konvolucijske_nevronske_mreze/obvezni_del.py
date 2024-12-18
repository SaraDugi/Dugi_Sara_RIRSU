import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, InputLayer, Dropout, BatchNormalization
from keras._tf_keras.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

data_dir = r"Dugi_Sara_RIRSU/2_klasifikacija/shapes"
save_dir = r"Dugi_Sara_RIRSU/5_konvolucijske_nevronske_mreze"
base_model_dir = os.path.join(save_dir, "base_model")
optimized_model_dir = os.path.join(save_dir, "optimized_model")
comparison_dir = os.path.join(save_dir, "comparison")
categories = ['circles', 'squares', 'triangles']

def create_dataframe(data_dir, categories):
    filepaths = []
    labels = []
    for category in categories:
        folder = os.path.join(data_dir, category)
        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                filepaths.append(os.path.join(folder, filename))
                labels.append(category)
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

full_df = create_dataframe(data_dir, categories)
train_df, test_df = train_test_split(full_df, test_size=0.2, stratify=full_df['label'], random_state=1234)
os.makedirs(save_dir, exist_ok=True)

train_csv_path = os.path.join(save_dir, "train_data.csv")
test_csv_path = os.path.join(save_dir, "test_data.csv")
train_df.to_csv(train_csv_path, index=False)
test_df.to_csv(test_csv_path, index=False)

# normalizacija slik
train_data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_data_gen = ImageDataGenerator(rescale=1./255)

# pridobitev iteratorjev
def get_data_generators(train_df, test_df, input_shape=(28, 28), batch_size=32):
    train_generator = train_data_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=input_shape,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=1234,
        subset='training'
    )
    val_generator = train_data_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filepath',
        y_col='label',
        target_size=input_shape,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=batch_size,
        shuffle=True,
        seed=1234,
        subset='validation'
    )
    test_generator = test_data_gen.flow_from_dataframe(
        dataframe=test_df,
        x_col='filepath',
        y_col='label',
        target_size=input_shape,
        color_mode='grayscale',
        class_mode='categorical',
        batch_size=1,
        shuffle=False
    )
    return train_generator, val_generator, test_generator

# osnovni model CNN
def build_model(input_shape=(28, 28, 1), learning_rate=0.001):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(16, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Optimiziran model CNN
def build_optimized_model(input_shape=(28, 28, 1), learning_rate=0.0005):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        AveragePooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_metrics(history, fold, model_name, save_dir):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Učna množica')
    plt.plot(history.history['val_accuracy'], label='Validacijska množica')
    plt.title(f'Točnost (fold {fold}) - {model_name}')
    plt.xlabel('Epoh')
    plt.ylabel('Točnost')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Učna množica')
    plt.plot(history.history['val_loss'], label='Validacijska množica')
    plt.title(f'Izguba (fold {fold}) - {model_name}')
    plt.xlabel('Epoh')
    plt.ylabel('Izguba')
    plt.legend()

    plt.tight_layout()
    
    specific_save_dir = base_model_dir if model_name == "Base_Model" else optimized_model_dir
    plt.savefig(os.path.join(specific_save_dir, f'{model_name}_metrics_fold_{fold}.png'))
    plt.close()

# 5-kratna prečna validacija
def train_and_evaluate_models(train_df, test_df, save_dir, input_shape=(28, 28), epochs=50):
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)

    # Train Base Model
    print("\nTraining Base Model...")
    fold = 1
    for train_indices, val_indices in kf.split(train_df):
        print(f"\n----- Base Model - Fold {fold} -----")
        train_fold = train_df.iloc[train_indices]
        val_fold = train_df.iloc[val_indices]

        train_generator, val_generator, test_generator = get_data_generators(
            train_fold, test_df, input_shape=input_shape, batch_size=32
        )

        model = build_model(input_shape + (1,))
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=1
        )
        model.save(os.path.join(base_model_dir, f'base_model_fold_{fold}.keras'))
        plot_metrics(history, fold, "Base_Model", save_dir)
        fold += 1

    # Train Optimized Model
    print("\nTraining Optimized Model...")
    fold = 1
    for train_indices, val_indices in kf.split(train_df):
        print(f"\n----- Optimized Model - Fold {fold} -----")
        train_fold = train_df.iloc[train_indices]
        val_fold = train_df.iloc[val_indices]

        train_generator, val_generator, test_generator = get_data_generators(
            train_fold, test_df, input_shape=input_shape, batch_size=16
        )

        model = build_optimized_model(input_shape + (1,), learning_rate=0.0005)
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            verbose=1
        )
        model.save(os.path.join(optimized_model_dir, f'optimized_model_fold_{fold}.keras'))
        plot_metrics(history, fold, "Optimized_Model", save_dir)
        fold += 1

# trening obeh modelov
train_and_evaluate_models(
    train_df=train_df,
    test_df=test_df,
    save_dir=save_dir,
    input_shape=(28, 28),
    epochs=50
)

# ovrednotenje modelov na testnem naboru
def evaluate_model_on_test(model, test_generator):
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    return {
        'Accuracy': accuracy,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

def save_and_plot_combined_metrics(base_results, optimized_results, save_dir):
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall']

    combined_data = {
        metric: [
            [result[metric] for result in base_results],
            [result[metric] for result in optimized_results]
        ] for metric in metrics
    }

    for metric, data in combined_data.items():
        plt.figure()
        plt.boxplot(data, tick_labels=['Base_Model', 'Optimized_Model'])
        plt.title(f'{metric} Comparison')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.savefig(os.path.join(comparison_dir, f'{metric}_Comparison_Boxplot.png'))
        plt.close()

def compare_models(base_results, optimized_results, save_dir):
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
    base_avg = {metric: np.mean([result[metric] for result in base_results]) for metric in metrics}
    optimized_avg = {metric: np.mean([result[metric] for result in optimized_results]) for metric in metrics}

    comparison_df = pd.DataFrame({'Base_Model': base_avg, 'Optimized_Model': optimized_avg}).T
    comparison_df.to_csv(os.path.join(comparison_dir, 'Comparison_Metrics.csv'), index=True)

    comparison_df.plot(kind='bar', figsize=(10, 6), title='Comparison of Average Metrics')
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.savefig(os.path.join(comparison_dir, 'Comparison_Metrics_Barplot.png'))
    plt.close()

def train_evaluate_compare(train_df, test_df, save_dir, input_shape=(28, 28), epochs=50):
    kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    base_model_results = []
    optimized_model_results = []

    # Treniranje in ovrednotenje
    for fold, (train_indices, val_indices) in enumerate(kf.split(train_df), start=1):
        print(f"\nFold {fold}...")

        train_fold = train_df.iloc[train_indices]
        val_fold = train_df.iloc[val_indices]

        train_generator, val_generator, test_generator = get_data_generators(
            train_fold, test_df, input_shape=input_shape, batch_size=32
        )

        # Osnovni model
        print("Training Base Model...")
        base_model = build_model(input_shape + (1,))
        base_model.fit(train_generator, validation_data=val_generator, epochs=epochs, verbose=1)
        base_model.save(os.path.join(save_dir, f'base_model_fold_{fold}.keras'))
        base_metrics = evaluate_model_on_test(base_model, test_generator)
        base_model_results.append(base_metrics)

        # Optimiziran model
        print("Training Optimized Model...")
        optimized_model = build_optimized_model(input_shape + (1,), learning_rate=0.0005)
        optimized_model.fit(train_generator, validation_data=val_generator, epochs=epochs, verbose=1)
        optimized_model.save(os.path.join(save_dir, f'optimized_model_fold_{fold}.keras'))
        optimized_metrics = evaluate_model_on_test(optimized_model, test_generator)
        optimized_model_results.append(optimized_metrics)

    save_and_plot_combined_metrics(base_model_results, optimized_model_results, save_dir)
    compare_models(base_model_results, optimized_model_results, save_dir)

train_evaluate_compare(
    train_df=train_df,
    test_df=test_df,
    save_dir=save_dir,
    input_shape=(28, 28),
    epochs=50
)