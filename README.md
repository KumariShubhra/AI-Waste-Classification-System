# ♻️ AI Waste Classification System

## 📌 Overview
This project is a Flask-based web application that classifies waste into categories like metal, organic, paper, plastic, and others using a trained deep learning model.

## 📂 Dataset

This project uses an image dataset for waste classification, containing categories such as:
- Metal
- Organic
- Paper
- Plastic
- Other

A small sample of the dataset is included in this repository for demonstration purposes.

The complete dataset consists of multiple images for training and testing the model. Due to size limitations, the full dataset is not included in this repository.

However, similar waste classification datasets are easily available on Kaggle. You can search for:
"Garbage Classification Dataset" or "Waste Classification Dataset"

Download the dataset and place it inside the `Dataset/` folder before running the project.

## 🚀 Features
- Upload waste image
- Predict waste category
- Real-time results
- Simple UI

## 🛠️ Tech Stack
- Python
- Flask
- TensorFlow
- NumPy
- Pillow

## 📂 Project Structure
- app.py → main backend
- templates/ → HTML files
- static/ → CSS & JS
- model (.h5) → trained CNN model

## ▶️ How to Run
pip install -r requirements.txt  
python app.py

## 📸 Output

### 🏠 Home Interface
User-friendly interface to upload waste images.
![Home](screenshots/home_page.png)

### 🤖 Prediction Result
The model predicts the waste category with probabilities.
![Result](screenshots/result.png)

## 🧪 Test Example

### Input Image
![Test Input](screenshots/test_image.jpg)
![Test Input](screenshots/test_image2.jpg)


## 🔮 Future Scope
- Real-time camera detection  
- Mobile app integration  
- IoT-based smart bins
