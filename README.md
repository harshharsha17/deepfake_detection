
# 🛡️ Deepfake Detection

This project is a Deepfake Detection system built using a Convolutional Neural Network (CNN) model. It allows users to upload an image and predict whether the image is real or fake.

---

## 🔍 Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

---

## 📂 Project Structure

```bash
deepfake_detection/
|— app.py                 # Flask web application
|— final_deepfake_detector.h5  # First trained CNN model
|— final_deepfake_detector2.h5 # Second trained CNN model (alternate)
|— predict.py             # Script for making predictions
|— requirements.txt       # Python dependencies
|— train.py               # Script to train the CNN model
|— templates/
    |— index.html          # Frontend template for the web app
```

---

## 🔍 Features

- 📤 Upload an image through a web interface.
- 🔍 Predict if the uploaded image is a real or fake face.
- 📊 Use a trained CNN model for accurate predictions.

---

## 🚀 Installation

1. **Clone** the repository or **download** the ZIP.
2. Navigate into the project directory:

```bash
cd deepfake_detection
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔄 How to Run

### Train the Model (Optional)

```bash
python train.py
```

### Run the Web Application

```bash
python app.py
```

Then open your browser and go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

---

## 📄 Files Description

- **app.py**: Flask server to handle uploads and make predictions.
- **predict.py**: Functions for loading the model and predicting real/fake images.
- **train.py**: Script to train the CNN on your dataset.
- **final_deepfake_detector.h5**: Pre-trained model for prediction.
- **index.html**: Simple frontend form for image uploads.

---

## 💪 Requirements

Main dependencies:
- Flask
- TensorFlow
- Keras
- OpenCV
- NumPy

(Full list in `requirements.txt`)

---

## 📈 Notes

- For faster model training, using a GPU is highly recommended.
- The model expects preprocessed and resized images (handled inside the code).

---

## 🌐 License

This project is for educational purposes only.
