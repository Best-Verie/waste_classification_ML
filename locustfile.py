
from locust import HttpUser, task, between
from PIL import Image
import io

# Create test image
img = Image.new('RGB', (224, 224), color='gray')
buf = io.BytesIO()
img.save(buf, format='JPEG')
TEST_IMG = buf.getvalue()

class User(HttpUser):
    wait_time = between(1, 3)
    
    @task(1)
    def health(self):
        self.client.get("/health")
    
    @task(1)
    def metrics(self):
        self.client.get("/metrics")
    
    @task(5)
    def predict(self):
        self.client.post("/predict", files={"file": ("test.jpg", TEST_IMG, "image/jpeg")})
