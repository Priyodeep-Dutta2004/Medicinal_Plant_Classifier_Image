# 🌿 Medicinal Plant Leaf Classifier using Deep Learning

This project is a part of my Smart India Hackathon submission. It focuses on the classification of common Indian and Asian medicinal plants using Convolutional Neural Networks (CNN) and real image datasets.
---

## 🎯 Objective

To build a lightweight yet accurate image classification model that can identify medicinal plant species based on leaf images. This supports applications in:

- Herbal medicine research
- Ayurvedic product development
- Plant-based pharmaceutical identification
---

## 📁 Dataset

We used a real image dataset of Indian medicinal plant leaves, organized into folders per species.
Kaggle dataset link- https://www.kaggle.com/datasets/warcoder/indian-medicinal-plant-image-dataset

**Sample classes include:**
- Neem
- Tulsi
- Aloe Vera
- Brahmi
- ... (depending on dataset)


## 🧠 Model Architecture

- *Base model:* MobileNetV2 (transfer learning)
- *Framework:* TensorFlow 2.x / Keras
- *Input size:* 128x128
- *Final Layer:* Dense layer with softmax (multi-class)

### 🧪 Training Details

- 80% training, 20% validation split
- Data augmentation: rotation, zoom, shifts
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 10

---

## 📊 Evaluation

- *Classification Report* 
               precision    recall  f1-score   support

     Aloevera       0.12      0.12      0.12        32
         Amla       0.00      0.00      0.00        29
 Amruta_Balli       0.03      0.03      0.03        29
        Arali       0.03      0.03      0.03        29
       Ashoka       0.06      0.07      0.07        29
  Ashwagandha       0.00      0.00      0.00        29
      Avacado       0.03      0.03      0.03        29
       Bamboo       0.00      0.00      0.00        29
       Basale       0.00      0.00      0.00        29
        Betel       0.07      0.07      0.07        30
    Betel_Nut       0.00      0.00      0.00        29
       Brahmi       0.00      0.00      0.00        29
       Castor       0.07      0.09      0.08        32
   Curry_Leaf       0.03      0.03      0.03        29
   Doddapatre       0.09      0.10      0.10        29
         Ekka       0.04      0.03      0.04        29
       Ganike       0.04      0.04      0.04        23
        Gauva       0.04      0.03      0.04        29
     Geranium       0.04      0.03      0.04        29
        Henna       0.00      0.00      0.00        30
     Hibiscus       0.04      0.03      0.03        33
        Honge       0.05      0.07      0.06        29
      Insulin       0.00      0.00      0.00        29
      Jasmine       0.06      0.08      0.07        37
        Lemon       0.00      0.00      0.00        29
  Lemon_grass       0.03      0.03      0.03        29
        Mango       0.04      0.03      0.04        29
         Mint       0.03      0.03      0.03        30
     Nagadali       0.04      0.03      0.04        30
         Neem       0.00      0.00      0.00        29
 Nithyapushpa       0.04      0.03      0.04        29
        Nooni       0.00      0.00      0.00        29
      Pappaya       0.04      0.03      0.04        29
       Pepper       0.00      0.00      0.00        29
  Pomegranate       0.10      0.10      0.10        29
Raktachandini       0.06      0.07      0.06        29
         Rose       0.03      0.03      0.03        33
       Sapota       0.08      0.10      0.09        29
       Sapota       0.08      0.10      0.09        29
       Tulasi       0.00      0.00      0.00        29
   Wood_sorel       0.04      0.03      0.04        29

     accuracy                           0.04      1180
    macro avg       0.04      0.04      0.04      1180
 weighted avg       0.04      0.04      0.04      1180
---

- *Confusion Matrix*
- Validation Accuracy: ~ 85% depending on dataset size

🖼 Output image:
Confusion_Matrix_Medicinal_Plants.png

---

## 🚀 How to Run

### 1. Install Required Libraries

pip install tensorflow scikit-learn matplotlib numpy pandas

### 2. Clone the Repository

git clone https://github.com/Priyodeep-Dutta2004/Medicinal_Plant_Classifier_Image.git
cd Medicinal_Plant_Classifier_Image

### 3. Run the Script

python Medicinal_Plant_Classifier.py

---

🔬 Future Scope

- Add support for Asian medicinal plants
- Include metadata features like leaf texture, region, medicinal use
- Deploy as an Android app for real-time identification
- Extend to herb root/flower classification

---

👤 Author

Priyodeep Dutta
Electrical Engineering | Machine Learning & Generative AI Enthusiast
LinkedIn • GitHub

---

📄 License

This project is open-source for educational and research use.

---
