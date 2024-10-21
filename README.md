# ASL-Gesture-Recognition
Welcome to the ASL to Speech Recognition System! This project leverages machine learning techniques to translate American Sign Language (ASL) gestures into spoken words in real-time. Utilizing TensorFlow's deep learning capabilities, MediaPipe for hand landmark detection, and OpenCV for video processing, this system aims to bridge the communication gap for the Deaf and Hard of Hearing community.

Features

Real-Time Gesture Recognition: Instantly translate ASL gestures captured via a webcam into spoken words or phrases.
Bidirectional LSTM Model: Utilizes temporal patterns in gesture sequences for accurate classification.
MediaPipe Integration: Efficiently detects and processes hand landmarks for gesture analysis.
User-Friendly Interface: Simple controls to switch between recognition modes and operate the application seamlessly.
Installation

Prerequisites
Python 3.7 or higher
Webcam

Steps

Clone the Repository:
Navigate to your desired directory and clone the project:

bash
Copy code
git clone https://github.com/yourusername/asl-to-speech-recognition.git
cd asl-to-speech-recognition
Create a Virtual Environment (Optional):
bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
bash
Copy code
pip install -r requirements.txt
If requirements.txt is unavailable, install manually:

bash
Copy code
pip install numpy scikit-learn tensorflow mediapipe opencv-python matplotlib seaborn
Data Preparation

Organize your dataset as follows:

css
Copy code
ASL_to_Speech_Recognition/
├── ASL_Data_Live/
│   ├── A/
│   │   ├── gesture1.npy
│   │   └── ...
│   ├── B/
│   └── ...
├── ASL_Data_Images/
│   ├── A/
│   │   ├── frame1.npy
│   │   └── ...
│   ├── B/
│   └── ...
├── model_training.py
├── real_time_recognition.py
├── requirements.txt
└── README.md
ASL_Data_Live: Contains sequences of hand landmarks for each ASL gesture.
ASL_Data_Images: Contains individual frames of hand landmarks, which are converted into sequences during preprocessing.
Ensure that each .npy file is correctly labeled and follows the specified shape requirements.

Model Training

Overview
The model_training.py script processes the collected data, handles class imbalances, normalizes features, trains a Bidirectional LSTM-based neural network, and saves the trained model along with necessary preprocessing tools.

Steps
Run the Training Script:
bash
Copy code
python model_training.py
Outputs:
Upon successful training, the following files will be saved:

asl_combined_model.keras: Final trained model.
best_asl_model.keras: Best-performing model based on validation loss.
label_encoder.pkl: Encodes class labels.
scaler.pkl: Normalizes input features.
Evaluation:
The script generates training history plots, a classification report, and a confusion matrix to assess model performance.
Real-Time Recognition

Overview
The real_time_recognition.py script utilizes the trained model to recognize ASL gestures captured via a webcam in real-time, translating them into spoken words or phrases.

Steps
Ensure Required Files Are Present:
asl_combined_model.keras
best_asl_model.keras
label_encoder.pkl
scaler.pkl
Run the Recognition Script:
bash
Copy code
python real_time_recognition.py
Operating the Application:
Select Mode: Choose between Letter, Word, or Phrase recognition.
Perform Gestures: Sign within the webcam's view.
Controls:
Press 'q': Quit the application.
Press 'm': Switch recognition modes.
Usage Instructions

Start the Application:
bash
Copy code
python real_time_recognition.py
Select Recognition Mode:
1: Letter Recognition
2: Word Recognition
3: Phrase Recognition
Perform Gestures:
Position your hand within the webcam frame and perform the desired ASL gesture.
The recognized gesture will display on the screen.
Switch Modes or Exit:
Press 'm': To change modes.
Press 'q': To exit the application.
Project Structure

bash
Copy code
ASL_to_Speech_Recognition/
├── ASL_Data_Live/           # Live gesture sequences
├── ASL_Data_Images/         # Individual gesture frames
├── model_training.py        # Script to train the model
├── real_time_recognition.py # Script for real-time recognition
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
Dependencies

NumPy
Scikit-Learn
TensorFlow
MediaPipe
OpenCV
Matplotlib
Seaborn
Install all dependencies using:

bash
Copy code
pip install -r requirements.txt
Troubleshooting

Common Issues
Missing label_encoder.pkl:
Solution: Ensure that model_training.py was run successfully and that label_encoder.pkl was saved.
Webcam Access Errors:
Solution: Check webcam connection and permissions. Ensure no other application is using the webcam.
Low Recognition Accuracy:
Solution:
Ensure ample and balanced training data.
Operate in well-lit environments.
Maintain consistent hand positioning.
Module Not Found Errors:
Solution: Verify all dependencies are installed. Reinstall missing packages using pip.
Additional Help
For issues not covered here, consult the documentation of the respective libraries or seek support through relevant forums.

Contact

For questions, support, or contributions, please reach out:

Email: beka.qendrim1@gmail.com
LinkedIn: https://www.linkedin.com/in/qendrim-beka-35b6792a1/

