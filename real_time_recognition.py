# real_time_recognition.py
# 1. Imported essential libraries like OpenCV for video capture, TensorFlow for model loading, Mediapipe for hand landmark detection, and Scikit-learn's StandardScaler for data scaling.
# 2. Defined a function to load the pre-trained model and scaler from disk for real-time gesture recognition.
# 3. Initialized Mediapipe’s hand detection module to capture hand landmarks from a live webcam feed.
# 4. For each frame captured from the webcam, processed the hand landmarks to extract key points that represent hand gestures.
# 5. Scaled the landmark data using the loaded StandardScaler to match the format the model was trained on.
# 6. Made predictions on the live data using the loaded TensorFlow model to recognize the current hand gesture.
# 7. Displayed the real-time predictions on the video feed using OpenCV.
# 8. Provided functionality to stop or pause the real-time recognition using keyboard input (e.g., pressing 'q' to quit).

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import os

# ==================== Initialization ====================

def load_resources(model_path='asl_combined_model.keras',
                  scaler_path='scaler.pkl'):
    """
    Load the trained model and scaler.
    
    Parameters:
    - model_path (str): Path to the trained Keras model.
    - scaler_path (str): Path to the saved StandardScaler.
    
    Returns:
    - model (tf.keras.Model): Loaded Keras model.
    - scaler (StandardScaler): Loaded StandardScaler.
    """
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from '{model_path}'.")
    
    # Check if scaler exists
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file '{scaler_path}' not found.")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"Loaded scaler from '{scaler_path}'.")
    
    return model, scaler

def initialize_mediapipe():
    """
    Initialize MediaPipe Hands.
    
    Returns:
    - hands (mp.solutions.hands.Hands): Initialized MediaPipe Hands object.
    - mp_draw (mp.solutions.drawing_utils.DrawingUtils): MediaPipe drawing utilities.
    """
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,       # For live video
        max_num_hands=1,               # Single hand
        min_detection_confidence=0.7,  # Detection confidence threshold
        min_tracking_confidence=0.5    # Tracking confidence threshold
    )
    return hands, mp_draw

def select_mode():
    """
    Prompt the user to select the recognition mode.
    
    Returns:
    - mode (str): Selected mode ('Letter', 'Word', or 'Phrase').
    """
    print("\nASL Recognition Modes:")
    print("1. Letter Recognition")
    print("2. Word Recognition")
    print("3. Phrase Recognition")
    print("Select a mode by entering the corresponding number.")
    while True:
        mode_input = input("Enter mode (1/2/3): ").strip()
        if mode_input == '1':
            return 'Letter'
        elif mode_input == '2':
            return 'Word'
        elif mode_input == '3':
            return 'Phrase'
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

def preprocess_sequence(sequence, scaler):
    """
    Normalize the sequence using the provided scaler.
    
    Parameters:
    - sequence (list): List of landmark coordinates.
    - scaler (StandardScaler): Fitted scaler object.
    
    Returns:
    - sequence_scaled (np.ndarray): Normalized and reshaped sequence.
    """
    sequence_array = np.array(sequence, dtype=np.float32)  # Shape: (sequence_length, num_features)
    sequence_scaled = scaler.transform(sequence_array)      # Normalize
    return sequence_scaled

def recognize_gesture(model, scaler, sequence, classes):
    """
    Recognize the gesture based on the sequence.
    
    Parameters:
    - model (tf.keras.Model): Trained Keras model.
    - scaler (StandardScaler): Fitted scaler for normalization.
    - sequence (list): List of landmark coordinates.
    - classes (list): List of class labels.
    
    Returns:
    - prediction_label (str): Predicted gesture label.
    """
    # Preprocess the sequence
    sequence_scaled = preprocess_sequence(sequence, scaler)  # Shape: (sequence_length, num_features)
    
    # Reshape for model input: (1, sequence_length, num_features)
    input_data = sequence_scaled.reshape(1, sequence_scaled.shape[0], sequence_scaled.shape[1])
    
    # Make prediction
    prediction = model.predict(input_data)
    class_id = np.argmax(prediction, axis=1)[0]
    
    # Map class_id to class label
    prediction_label = classes[class_id]
    
    return prediction_label

def main():
    # ==================== Load Resources ====================
    
    try:
        model, scaler = load_resources()
    except FileNotFoundError as e:
        print(e)
        return
    except Exception as e:
        print(f"Unexpected error during resource loading: {e}")
        return
    
    # ==================== Initialize MediaPipe ====================
    
    hands, mp_draw = initialize_mediapipe()
    print("Initialized MediaPipe Hands.")
    
    # ==================== Video Capture ====================
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return
    print("Webcam access successful.")
    
    # ==================== Mode Selection ====================
    
    mode = select_mode()
    print(f"\nMode selected: {mode} Recognition")
    print("Press 'q' to quit.")
    print("Press 'm' to switch modes.")
    
    # ==================== Variables Initialization ====================
    
    num_features = 63       # 21 landmarks * 3 coordinates
    sequence_length = 30    # Must match the training sequence length
    landmark_sequence = []  # To store landmarks over time
    
    recognized_text = ''
    sentence = []            # Used in Phrase mode
    
    # Define the list of classes manually
    classes = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
        'Y', 'Z'
    ]
    
    # ==================== Main Loop ====================
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame.")
            break
        
        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        result = hands.process(img_rgb)
        
        # Initialize prediction label
        prediction_label = ''
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                
                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Append landmarks to the sequence
                landmark_sequence.append(landmarks)
                
                if len(landmark_sequence) == sequence_length:
                    # Recognize the gesture
                    try:
                        prediction_label = recognize_gesture(model, scaler, landmark_sequence, classes)
                    except Exception as e:
                        print(f"Error during gesture recognition: {e}")
                        prediction_label = "Error"
                    
                    # Update recognized text based on the mode
                    if mode == 'Letter':
                        recognized_text = prediction_label
                    elif mode == 'Word':
                        recognized_text = prediction_label
                    elif mode == 'Phrase':
                        # Append the recognized word to the sentence
                        sentence.append(prediction_label)
                        recognized_text = ' '.join(sentence)
                    
                    # Reset the sequence for the next prediction
                    landmark_sequence = []
        
        else:
            # No hand detected; reset the sequence
            landmark_sequence = []
        
        # Display mode-specific outputs
        if mode in ['Letter', 'Word']:
            display_text = recognized_text
        elif mode == 'Phrase':
            display_text = recognized_text
        else:
            display_text = ''
        
        # Overlay the recognized text on the frame
        cv2.putText(frame, f'Mode: {mode}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Recognized: {display_text}', (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('ASL Recognition', frame)
        
        # Handle keyboard inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Quit the program
            print("\nExiting ASL Recognition.")
            break
        elif key == ord('m'):
            # Switch modes
            mode = select_mode()
            print(f"\nMode switched to: {mode} Recognition")
            # Reset variables
            recognized_text = ''
            sentence = []
            landmark_sequence = []
    
    # ==================== Cleanup ====================
    
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("Released webcam and closed all windows.")

if __name__ == "__main__":
    main()
