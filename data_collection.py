# data_collection.py

import os
import cv2
import mediapipe as mp
import numpy as np

# ==================== Initialization ====================

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Define data paths
LIVE_DATA_PATH = os.path.join('ASL_Data_Live')
IMAGE_DATA_PATH = os.path.join('ASL_Data_Images')

# Create directories if they don't exist
os.makedirs(LIVE_DATA_PATH, exist_ok=True)
os.makedirs(IMAGE_DATA_PATH, exist_ok=True)

# ==================== Mode Selection ====================

def select_data_collection_mode():
    """
    Displays a prompt to select the data collection mode.
    Returns the selected mode as a string: 'Image' or 'Live'.
    """
    print("\nData Collection Modes:")
    print("1. Process Image Dataset")
    print("2. Collect Live Data")
    print("Select a mode by entering the corresponding number.")
    while True:
        mode_input = input("Enter mode (1/2): ")
        if mode_input == '1':
            return 'Image'
        elif mode_input == '2':
            return 'Live'
        else:
            print("Invalid input. Please enter 1 or 2.")

# Select the data collection mode
mode = select_data_collection_mode()
print(f"\nMode selected: {mode} Data Collection")

# ==================== Image Dataset Processing ====================

if mode == 'Image':
    # Initialize MediaPipe Hands for static images
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5  # Adjusted for better detection
    ) as hands:
        # Path to the image dataset
        DATASET_PATH = input("Enter the path to the image dataset: ").strip()
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"Dataset path '{DATASET_PATH}' does not exist.")

        # Get all labels (folders in the dataset directory)
        labels = [label for label in os.listdir(DATASET_PATH)
                  if os.path.isdir(os.path.join(DATASET_PATH, label))]
        print(f"Labels found: {labels}")

        if not labels:
            raise ValueError(f"No label folders found in '{DATASET_PATH}'. Ensure your dataset is organized correctly.")

        for label in labels:
            label_path = os.path.join(DATASET_PATH, label)
            save_label_path = os.path.join(IMAGE_DATA_PATH, label)
            os.makedirs(save_label_path, exist_ok=True)

            print(f"\nProcessing label: {label}")

            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                
                # Skip non-image files
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    print(f"Skipping non-image file: {image_path}")
                    continue

                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image {image_path}. Skipping.")
                    continue

                # Check the number of channels
                if len(image.shape) == 2 or image.shape[2] == 1:
                    print(f"Image is grayscale: {image_path}. Converting to BGR.")
                    # Convert grayscale to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                elif image.shape[2] == 4:
                    print(f"Image has alpha channel: {image_path}. Converting to BGR.")
                    # Convert BGRA to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                # Convert to RGB
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                result = hands.process(img_rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # Extract landmarks
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        # Convert to numpy array
                        landmarks = np.array(landmarks, dtype=np.float32)
                        # Save landmarks
                        base_name = os.path.splitext(image_name)[0]
                        npy_path = os.path.join(save_label_path, f'{base_name}.npy')
                        np.save(npy_path, landmarks)
                        print(f"Saved landmarks for {image_path} as {npy_path}")
                    
                    # Optional: Visualize the detection
                    annotated_image = image.copy()
                    mp_draw.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.imshow('Hand Detection', annotated_image)
                    cv2.waitKey(1)  # Display each image briefly
                else:
                    print(f"No hand detected in {image_path}")

    # Close any OpenCV windows
    cv2.destroyAllWindows()
    print("\nImage data processing complete.")

# ==================== Live Data Collection ====================

elif mode == 'Live':
    # Initialize MediaPipe Hands for live capture
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    ) as hands:
        # Video capture
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("Cannot open webcam. Please ensure it is connected and accessible.")

        # Collect data for letters, words, or phrases
        gesture_type = input("Enter the type of gesture (Letter/Word/Phrase): ").strip().capitalize()
        assert gesture_type in ['Letter', 'Word', 'Phrase'], "Invalid gesture type. Please enter 'Letter', 'Word', or 'Phrase'."

        # Get the label for the gesture
        label = input(f"Enter the label for the {gesture_type}: ").strip()
        if not label:
            raise ValueError("Label cannot be empty.")

        # Create directory for the label
        label_path = os.path.join(LIVE_DATA_PATH, label)
        os.makedirs(label_path, exist_ok=True)

        sequence_length = 30  # Number of frames per sequence
        num_sequences = 50    # Number of sequences to collect

        print("\nStarting live data collection...")
        print("Press 's' to start recording a sequence.")
        print("Press 'q' to quit.")

        sequence_count = 0
        collecting = False
        frame_count = 0
        sequence = []

        while sequence_count < num_sequences:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam. Exiting.")
                break

            # Flip and convert the frame
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    # Draw landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    if collecting:
                        sequence.append(landmarks)
                        frame_count += 1

                        if frame_count == sequence_length:
                            # Convert to numpy array
                            sequence_array = np.array(sequence, dtype=np.float32)
                            # Save the sequence
                            npy_path = os.path.join(label_path, f'{gesture_type}_{sequence_count}.npy')
                            np.save(npy_path, sequence_array)
                            sequence_count += 1
                            print(f"Collected sequence {sequence_count}/{num_sequences}", end='\r')
                            # Reset variables
                            sequence = []
                            frame_count = 0
                            collecting = False

            # Display instructions
            cv2.putText(frame, f'Collecting: {collecting}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f'Sequences Collected: {sequence_count}/{num_sequences}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Live Data Collection', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                if not collecting:
                    collecting = True
                    frame_count = 0
                    sequence = []
                    print("\nStarted recording sequence.")
            elif key & 0xFF == ord('q'):
                print("\nExiting data collection.")
                break

        cap.release()
        cv2.destroyAllWindows()

        print("\nLive data collection complete.")

else:
    print("Invalid mode selected. Exiting.")
