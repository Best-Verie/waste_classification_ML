

# Garbage Classification ML Pipeline

A machine learning pipeline for classifying waste images into 12 categories using deep learning.

---

## Demo

**Video Demo:** [Watch Demo](https://share.vidyard.com/watch/Gcb9wMZnkHG3yxKDM9Lg2s)

**Live URL:** [https://waste-classifier-b8wx.onrender.com/](https://waste-classifier-b8wx.onrender.com/)
**Swagger:** [https://waste-classifier-b8wx.onrender.com/docs#](https://waste-classifier-b8wx.onrender.com/docs#)


---

## Dataset

**Source:** [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)

- **Images:** ~15,000
- **Classes:** 12 (battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass)
- **Split:** 80% Train / 20% Validation

---

## Technologies Used

| Category | Technologies |
|----------|--------------|
| ML | TensorFlow, Keras, MobileNetV2, Scikit-learn |
| Backend | FastAPI, Uvicorn |
| Testing | Locust |
| Deployment | Docker, Render |

---

## Model

- **Architecture:** MobileNetV2 (Transfer Learning)
- **Input:** 224x224 RGB image
- **Output:** 12 class probabilities
- **Accuracy:** ~85%

---

## Project Structure

```
waste_classification_ML/
├── index.py              # API + UI
├── locustfile.py         # Load testing
├── Dockerfile
├── requirements.txt
├── notebook/
│   └── garbage_classification.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── prediction.py
├── models/
│   ├── garbage_classifier.h5
│   └── class_mapping.json
└── data/
    ├── train/
    └── test/
```

---

## Installation

```bash
git clone https://github.com/Best-Verie/waste_classification_ML.git
cd waste_classification_ML
pip install -r requirements.txt
python index.py
```

Open: http://localhost:8000

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| GET | `/metrics` | Predictions count, uptime |
| POST | `/predict` | Classify image |
| POST | `/retrain` | Trigger retraining |

---

## Load Testing Results

Tested with Locust (10 users):

| Endpoint | Avg Response | Failures |
|----------|--------------|----------|
| /health | 46ms | 0% |
| /metrics | 89ms | 0% |
| /predict | 265ms | 0% |

![locust summary](image.png)
![locust charts](image-1.png)


---

## Deployment (Render)

1. Push code to GitHub
2. Go to [render.com](https://render.com)
3. Create new Web Service
4. Connect GitHub repo
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `python index.py`
7. Deploy

---

## Author

**Best Verie**  
African Leadership University  
November 2025
