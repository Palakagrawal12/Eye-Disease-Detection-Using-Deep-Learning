# Eye-Disease-Detection-Using-Deep-Learning
Project Overview
This project utilizes deep learning techniques to classify various eye diseases based on retinal images. The model is trained to recognize four categories:

Normal
Cataract
Diabetic Retinopathy
Glaucoma
A pre-trained Xception model with transfer learning is used for feature extraction and classification. The system is deployed as a Flask web application, allowing users to upload images and receive predictions.

Technologies Used
Python 3.x
TensorFlow & Keras (Deep Learning)
Flask (Web Framework)
NumPy, Pandas, Matplotlib (Data Processing & Visualization)
OpenCV & PIL (Image Processing)
Dataset & Preprocessing
The dataset consists of labeled retinal images categorized into four disease classes.
Images are resized to 299x299 to match the input requirements of the Xception model.
Normalization is applied to scale pixel values between 0 and 1.
Model Training
The model follows these steps:

Load & Preprocess Data
Configure ImageDataGenerator for Augmentation
Use Xception for Feature Extraction
Train Fully Connected Layers for Classification
Save the Trained Model (xception.h5)
Training Command:
python train_model.py --epochs 50 --batch_size 32 --learning_rate 0.001
Deployment with Flask
The trained model is served using Flask.
Users can upload retinal images, and the system predicts the disease category.
The web interface is designed for easy interaction.
Running the Web Application:
python app.py
Project Structure
Eye_Disease_Detection/
│── app.py                 # Flask Application
│── xception.h5            # Trained Model
│── templates/
│   └── index.html         # Frontend UI
│── static/uploads/        # Uploaded Images
│── Model_Training.ipynb   # Jupyter Notebook for Training
└── README.md              # Documentation
Future Enhancements
Increase dataset size for improved accuracy.
Implement Explainable AI (XAI) to provide insights into predictions.
Extend model support for additional eye diseases.
Deploy as a cloud-based API for broader accessibility.
This project demonstrates the application of AI in healthcare, providing a potential solution for early eye disease detection and diagnosis.
