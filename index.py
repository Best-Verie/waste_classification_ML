

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import numpy as np
from PIL import Image
import keras
import io, json, os
from datetime import datetime

app = FastAPI(title="Garbage Classification API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Global
model = None
idx_to_class = {}
start_time = datetime.now()
prediction_count = 0
retrain_status = {"status": "idle"}

def load_model():
    global model, idx_to_class
    
    # Load class mapping first
    if os.path.exists("models/class_mapping.json"):
        with open("models/class_mapping.json") as f:
            idx_to_class = {int(k): v for k, v in json.load(f).get('idx_to_class', {}).items()}
    
    num_classes = len(idx_to_class) if idx_to_class else 12
    
    # Rebuild model architecture (same as training)
    from keras.applications import MobileNetV2
    from keras import layers, models as k_models
    
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False
    
    model = k_models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Load weights only (not full model)
    for path in ["models/garbage_classifier.h5", "models/best_model.h5", "models/best_model.keras"]:
        if os.path.exists(path):
            try:
                model.load_weights(path)
                print(f" Weights loaded from: {path}")
                return
            except Exception as e:
                print(f"Failed {path}: {e}")
    
    print(" No weights loaded - model initialized with random weights")

@app.on_event("startup")
async def startup():
    load_model()

# ============== UI ==============
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Garbage Classification</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: linear-gradient(135deg, #11998e, #38ef7d); min-height: 100vh; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        h1 { color: white; text-align: center; margin-bottom: 20px; }
        .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 5px 20px rgba(0,0,0,0.2); }
        .card h2 { color: #11998e; margin-bottom: 15px; }
        .upload-area { border: 2px dashed #11998e; border-radius: 10px; padding: 30px; text-align: center; cursor: pointer; }
        .upload-area:hover { background: #f0fff0; }
        input[type="file"] { display: none; }
        .btn { background: linear-gradient(135deg, #11998e, #38ef7d); color: white; border: none; padding: 12px 25px; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.5; }
        #preview { max-width: 200px; margin: 15px auto; display: none; border-radius: 10px; }
        .result { background: #f0f0f0; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .result h3 { color: #11998e; }
        .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center; }
        .metric { background: #11998e; color: white; padding: 15px; border-radius: 5px; }
        .metric span { font-size: 1.5em; font-weight: bold; }
        footer { text-align: center; color: white; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Garbage Classification</h1>
        
        <div class="card">
            <h2> Classify Waste</h2>
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p> Click to upload garbage image</p>
            </div>
            <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
            <img id="preview">
            <div style="text-align: center; margin-top: 15px;">
                <button class="btn" id="predictBtn" onclick="predict()" disabled>Classify</button>
            </div>
            <div id="result"></div>
        </div>
        
        <div class="card">
            <h2> Retrain Model</h2>
            <p>Epochs: <input type="number" id="epochs" value="5" min="1" max="20" style="width: 60px; padding: 5px;"></p>
            <button class="btn" onclick="retrain()">Start Retraining</button>
            <div id="retrainResult"></div>
        </div>
        
        <div class="card">
            <h2> System Metrics</h2>
            <div class="metrics">
                <div class="metric"><span id="predictions">0</span><br>Predictions</div>
                <div class="metric"><span id="uptime">0s</span><br>Uptime</div>
                <div class="metric"><span id="status">-</span><br>Status</div>
            </div>
            <button class="btn" onclick="loadMetrics()" style="margin-top: 15px;">Refresh</button>
        </div>
        
        <footer>Garbage Classification | Best Verie | ALU 2025</footer>
    </div>

    <script>
        let selectedFile = null;

        function previewImage(e) {
            selectedFile = e.target.files[0];
            const preview = document.getElementById('preview');
            preview.src = URL.createObjectURL(selectedFile);
            preview.style.display = 'block';
            document.getElementById('predictBtn').disabled = false;
        }

        async function predict() {
            if (!selectedFile) return;
            document.getElementById('result').innerHTML = '<p>Classifying...</p>';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('result').innerHTML = `
                    <div class="result">
                        <h3> ${data.prediction}</h3>
                        <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
                        <p><strong>Top 5:</strong></p>
                        ${data.top_5.map((p, i) => `<p>${i+1}. ${p.class}: ${(p.confidence * 100).toFixed(2)}%</p>`).join('')}
                    </div>`;
            } catch (e) {
                document.getElementById('result').innerHTML = `<p style="color:red">Error: ${e.message}</p>`;
            }
        }

        async function retrain() {
            const epochs = document.getElementById('epochs').value;
            document.getElementById('retrainResult').innerHTML = '<p>Starting...</p>';
            try {
                const res = await fetch(`/retrain?epochs=${epochs}`, { method: 'POST' });
                const data = await res.json();
                document.getElementById('retrainResult').innerHTML = `<p style="color:green"> ${data.message}</p>`;
            } catch (e) {
                document.getElementById('retrainResult').innerHTML = `<p style="color:red">Error: ${e.message}</p>`;
            }
        }

        async function loadMetrics() {
            try {
                const res = await fetch('/metrics');
                const data = await res.json();
                document.getElementById('predictions').textContent = data.predictions;
                document.getElementById('uptime').textContent = Math.round(data.uptime_seconds) + 's';
                document.getElementById('status').textContent = data.retrain_status;
            } catch (e) { console.error(e); }
        }

        loadMetrics();
        setInterval(loadMetrics, 5000);
    </script>
</body>
</html>
"""

# ============== API ==============
@app.get("/health")
def health():
    return {"status": "healthy" if model else "degraded", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return {
        "predictions": prediction_count,
        "uptime_seconds": (datetime.now() - start_time).total_seconds(),
        "retrain_status": retrain_status["status"]
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global prediction_count
    if model is None:
        raise HTTPException(503, "Model not loaded")
    
    img = Image.open(io.BytesIO(await file.read())).convert('RGB').resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    
    preds = model.predict(img_array, verbose=0)[0]
    top_idx = np.argsort(preds)[-5:][::-1]
    prediction_count += 1
    
    return {
        "prediction": idx_to_class.get(top_idx[0], f"Class_{top_idx[0]}"),
        "confidence": float(preds[top_idx[0]]),
        "top_5": [{"class": idx_to_class.get(i, f"Class_{i}"), "confidence": float(preds[i])} for i in top_idx]
    }

def do_retrain(data_dir, epochs):
    global model, retrain_status
    retrain_status = {"status": "running"}
    try:
        from keras.src.legacy.preprocessing.image import ImageDataGenerator
        
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        train_gen = datagen.flow_from_directory(data_dir, target_size=(224,224), subset='training')
        val_gen = datagen.flow_from_directory(data_dir, target_size=(224,224), subset='validation')
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen)
        model.save("models/best_model.keras")
        
        retrain_status = {"status": "completed", "accuracy": float(history.history['val_accuracy'][-1])}
    except Exception as e:
        retrain_status = {"status": "failed", "error": str(e)}

@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks, epochs: int = 5, data_dir: str = "data/train"):
    if retrain_status["status"] == "running":
        raise HTTPException(400, "Retraining already in progress")
    background_tasks.add_task(do_retrain, data_dir, epochs)
    return {"message": "Retraining started", "check_status": "/metrics"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
