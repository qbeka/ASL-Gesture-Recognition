# model_training.py
# 1. Imported necessary libraries such as TensorFlow, Scikit-learn, NumPy, and Matplotlib for model building, data processing, and visualization.
# 2. Defined constants such as SEQUENCE_LENGTH, NUM_FEATURES, and hyperparameters like EPOCHS and BATCH_SIZE for model configuration.
# 3. Loaded datasets from NumPy `.npy` files and split them into training and validation sets using Scikit-learn's train_test_split.
# 4. Calculated class weights to handle class imbalance, which improves model accuracy for underrepresented classes.
# 5. Used Scikit-learn's StandardScaler to normalize the input data for better model training performance.
# 6. Built a sequential neural network model using TensorFlow/Keras, with multiple layers to handle the input feature sequences and output classifications.
# 7. Trained the model using the compiled architecture on the scaled training data, validating it on validation data.
# 8. Saved the trained model and the scaler for later use in real-time recognition tasks.
# 9. Visualized training history (accuracy and loss) to track model performance and improvements over epochs.
# 10. Evaluated the trained model on the validation set and generated a classification report and confusion matrix to assess performance.
# 11. Saved important model metrics and visualizations to understand the model's behavior.

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ==================== Data Loading ====================

def load_data(live_data_path, image_data_path, sequence_length=30, num_features=63):
    """
    Load and preprocess data from live sequences and image datasets.

    Parameters:
    - live_data_path (str): Path to live data sequences.
    - image_data_path (str): Path to image data frames.
    - sequence_length (int): Number of frames per sequence.
    - num_features (int): Number of features per frame.

    Returns:
    - X_combined (np.ndarray): Combined feature data.
    - y_combined (np.ndarray): Combined labels.
    """
    X_live = []
    y_live = []
    X_image = []
    y_image = []

    # Load live data (sequences)
    if os.path.exists(live_data_path) and os.listdir(live_data_path):
        print("Loading live data sequences...")
        for label in os.listdir(live_data_path):
            label_path = os.path.join(live_data_path, label)
            if not os.path.isdir(label_path):
                print(f"Skipping non-directory: {label_path}")
                continue
            for sequence_file in os.listdir(label_path):
                npy_path = os.path.join(label_path, sequence_file)
                try:
                    sequence = np.load(npy_path)
                    if sequence.shape != (sequence_length, num_features):
                        print(f"Skipping {npy_path}: Expected shape {(sequence_length, num_features)}, got {sequence.shape}")
                        continue
                    X_live.append(sequence)
                    y_live.append(label)
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    continue
    else:
        print(f"No live data found in '{live_data_path}'. Proceeding without live data.")

    # Load image data (single frames)
    if os.path.exists(image_data_path) and os.listdir(image_data_path):
        print("Loading image data frames...")
        for label in os.listdir(image_data_path):
            label_path = os.path.join(image_data_path, label)
            if not os.path.isdir(label_path):
                print(f"Skipping non-directory: {label_path}")
                continue
            for npy_file in os.listdir(label_path):
                npy_path = os.path.join(label_path, npy_file)
                try:
                    landmarks = np.load(npy_path)
                    if landmarks.shape != (num_features,):
                        print(f"Skipping {npy_path}: Expected shape {(num_features,)}, got {landmarks.shape}")
                        continue
                    # Convert single frame to a sequence by repeating it
                    sequence = np.tile(landmarks, (sequence_length, 1))
                    X_image.append(sequence)
                    y_image.append(label)
                except Exception as e:
                    print(f"Error loading {npy_path}: {e}")
                    continue
    else:
        print(f"No image data found in '{image_data_path}'. Proceeding without image data.")

    # Combine live and image data
    X_combined = np.array(X_live + X_image)
    y_combined = np.array(y_live + y_image)

    if X_combined.size == 0:
        raise ValueError("No valid sequences found in both live and image data paths. Please check your data collection process.")

    print(f"Total samples loaded: {X_combined.shape[0]}")
    print(f"Sequence length: {X_combined.shape[1]}")
    print(f"Number of features: {X_combined.shape[2]}")

    return X_combined, y_combined

# ==================== Label Encoding ====================

def encode_labels(y):
    """
    Encode string labels into integers.

    Parameters:
    - y (np.ndarray): Array of string labels.

    Returns:
    - y_encoded (np.ndarray): Array of integer labels.
    - le (LabelEncoder): Fitted label encoder.
    """
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le

# ==================== Data Preprocessing ====================

def preprocess_data(X_train, X_test):
    """
    Normalize the feature data using StandardScaler.

    Parameters:
    - X_train (np.ndarray): Training feature data.
    - X_test (np.ndarray): Testing feature data.

    Returns:
    - X_train_scaled (np.ndarray): Normalized training data.
    - X_test_scaled (np.ndarray): Normalized testing data.
    - scaler (StandardScaler): Fitted scaler object.
    """
    scaler = StandardScaler()
    # Reshape to 2D for scaling: (num_samples * sequence_length, num_features)
    num_samples_train, sequence_length, num_features = X_train.shape
    num_samples_test = X_test.shape[0]
    
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_test_reshaped = X_test.reshape(-1, num_features)
    
    # Fit scaler on training data and transform both training and testing data
    scaler.fit(X_train_reshaped)
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(num_samples_train, sequence_length, num_features)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(num_samples_test, sequence_length, num_features)
    
    return X_train_scaled, X_test_scaled, scaler

# ==================== Handle Class Imbalance ====================

def compute_class_weights(y_train, num_classes):
    """
    Compute class weights to handle class imbalance.

    Parameters:
    - y_train (np.ndarray): Array of integer labels for training data.
    - num_classes (int): Number of unique classes.

    Returns:
    - class_weights_dict (dict): Dictionary mapping class indices to weights.
    """
    class_weights_values = class_weight.compute_class_weight('balanced',
                                                             classes=np.arange(num_classes),
                                                             y=y_train)
    class_weights_dict = {i: class_weights_values[i] for i in range(num_classes)}
    return class_weights_dict

# ==================== Model Building ====================

def build_model(sequence_length, num_features, num_classes):
    """
    Build and compile the LSTM-based model.

    Parameters:
    - sequence_length (int): Number of frames per sequence.
    - num_features (int): Number of features per frame.
    - num_classes (int): Number of unique classes.

    Returns:
    - model (tf.keras.Model): Compiled Keras model.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(sequence_length, num_features)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    
    return model

# ==================== Training and Evaluation ====================

def train_model(model, X_train, y_train, X_val, y_val, class_weights, epochs=50, batch_size=32):
    """
    Train the Keras model with Early Stopping and Model Checkpointing.

    Parameters:
    - model (tf.keras.Model): Compiled Keras model.
    - X_train (np.ndarray): Training feature data.
    - y_train (np.ndarray): Training labels.
    - X_val (np.ndarray): Validation feature data.
    - y_val (np.ndarray): Validation labels.
    - class_weights (dict): Class weights for handling imbalance.
    - epochs (int): Maximum number of epochs.
    - batch_size (int): Batch size.

    Returns:
    - history (tf.keras.callbacks.History): Training history.
    """
    # Define callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('best_asl_model.keras', monitor='val_loss', save_best_only=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, checkpoint],
        class_weight=class_weights,
        verbose=1
    )
    
    return history

# ==================== Visualization ====================

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.

    Parameters:
    - history (tf.keras.callbacks.History): Training history.
    """
    plt.figure(figsize=(14,6))
    
    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Plot the confusion matrix.

    Parameters:
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.
    - classes (list): List of class names.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def classification_report_print(y_true, y_pred, classes):
    """
    Print the classification report.

    Parameters:
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.
    - classes (list): List of class names.
    """
    report = classification_report(y_true, y_pred, target_names=classes)
    print("Classification Report:\n", report)

# ==================== Main Function ====================

def main():
    # Define paths
    LIVE_DATA_PATH = os.path.join('ASL_Data_Live')
    IMAGE_DATA_PATH = os.path.join('ASL_Data_Images')
    
    # Parameters
    SEQUENCE_LENGTH = 30
    NUM_FEATURES = 63
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Load data
    try:
        X, y = load_data(LIVE_DATA_PATH, IMAGE_DATA_PATH, SEQUENCE_LENGTH, NUM_FEATURES)
    except ValueError as ve:
        print(f"Data Loading Error: {ve}")
        return
    except Exception as e:
        print(f"Unexpected Error during Data Loading: {e}")
        return
    
    # Encode labels
    y_encoded, le = encode_labels(y)
    classes = le.classes_
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    
    # Check class distribution
    counter = Counter(y_encoded)
    print("Class distribution:", counter)
    
    # Compute class weights
    class_weights = compute_class_weights(y_encoded, num_classes)
    print("Computed class weights:", class_weights)
    
    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # Preprocess data
    X_train_scaled, X_val_scaled, scaler = preprocess_data(X_train, X_val)
    print("Data normalization complete.")
    
    # Save the scaler for future use
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved as 'scaler.pkl'.")
    
    # Build the model
    model = build_model(SEQUENCE_LENGTH, NUM_FEATURES, num_classes)
    
    # Train the model
    history = train_model(model, X_train_scaled, y_train, X_val_scaled, y_val, class_weights, EPOCHS, BATCH_SIZE)
    
    # Save the trained model
    model.save('asl_combined_model.keras')
    print("Trained model saved as 'asl_combined_model.keras'.")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model on validation data
    y_val_pred_probs = model.predict(X_val_scaled)
    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
    
    # Classification Report
    classification_report_print(y_val, y_val_pred, classes)
    
    # Confusion Matrix
    plot_confusion_matrix(y_val, y_val_pred, classes)

if __name__ == "__main__":
    main()
