Here’s a real-world ML use case example with Python code for development and deployment using MLOps principles. 
The example is a Customer Churn Prediction System for a telecom company. 
The goal is to predict which customers are likely to churn (leave the service) and deploy the model as a REST API using Docker and Kubernetes.

Use Case: Customer Churn Prediction System
Goal: Predict customer churn and deploy the model as a scalable API using MLOps practices.

Here's a **complete CI/CD pipeline** for deploying a fraud detection model **to Kubernetes using GitHub Actions.** 🚀  

---

## **🔷 Workflow Overview**
1️⃣ **Develop & Train ML Model** → Save & containerize API (Docker).  
2️⃣ **Push Code to GitHub** → Triggers CI/CD pipeline automatically.  
3️⃣ **GitHub Actions Pipeline**:  
   ✅ **Builds Docker Image**  
   ✅ **Pushes Image to Docker Hub**  
   ✅ **Deploys it to Kubernetes**  

---

## **🔹 Step 1: Train ML Model & Expose as API**

Python Code for Training and Saving the Model
# churn_prediction_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump

# Load and preprocess data
data = pd.read_csv('customer_churn.csv')
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
data.dropna(inplace=True)

# Feature engineering
X = data.drop(['customerID', 'Churn'], axis=1)
X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical variables
y = data['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model
dump(model, 'churn_model.joblib')
print("Model trained and saved!")

## **🔹 Step 2: API Development 
Flask REST API for Predictions
This Flask API loads the trained **fraud detection model** and predicts transactions.

### **📌 `app.py` (Flask API)**
```python
# app.py
from flask import Flask, request, jsonify
from joblib import load
import logging
import pandas as pd

app = Flask(__name__)

# Load the model
model = load('churn_model.joblib')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Root route
@app.route('/')
def home():
    return "Welcome to the Customer Churn Prediction API! Use the /predict endpoint to make predictions."

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])
        input_data = pd.get_dummies(input_data, drop_first=True)  # Ensure same preprocessing as training
       
        # Ensure all feature names are present
        feature_names = model.feature_names_in_
        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0

        input_data = input_data[feature_names]

        prediction = model.predict(input_data)[0]
        return jsonify({'churn': bool(prediction)})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
```

---

## **🔹 Step 3: Dockerize the App**
Create a Docker image so the model API runs consistently anywhere.

### **📌 `Dockerfile`**
```dockerfile
# Base Image
FROM python:3.12.4-slim

WORKDIR /app
COPY requirements.txt .
COPY app.py .
COPY fraud_model.joblib .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
```

---

### **📌 Type  pip freeze > requirements.txt in terminal for creating `requirements.txt`**
```
Flask
pandas
scikit-learn
joblib
```

---

## **🔹 Step 4: Kubernetes Deployment Files**
To deploy in **Kubernetes (K8s)**, we define two files:  
1. **Deployment** – Manages running API containers.  
2. **Service** – Exposes the API over the network.

### **📌 `fraud-detection-deployment.yml`**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: fraud-detection
  template:
    metadata:
      labels:
        app: fraud-detection
    spec:
      containers:
      - name: fraud-detection
        image: <DOCKER_USERNAME>/fraud-detection-api:latest
        ports:
        - containerPort: 5000
```

### **📌 `fraud-detection-service.yml`**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-service
spec:
  selector:
    app: fraud-detection
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

---

## **🔹 Step 5: CI/CD Pipeline with GitHub Actions**
Every time code is pushed to `main`, GitHub Actions:  
✅ Builds the Docker Image  
✅ Pushes it to Docker Hub  
✅ Deploys it to Kubernetes  

### **📌 `.github/workflows/deploy.yml`**
```yaml
name: Deploy Fraud Detection API

on:
  push:
    branches:
      - main  # Trigger CI/CD on push to main branch

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout Code
      uses: actions/checkout@v3

    - name: Log in to DockerHub
      run: echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin

    - name: Build Docker Image
      run: |
        docker build -t fraud-detection-api .
        docker tag fraud-detection-api:latest ${{ secrets.DOCKER_USERNAME }}/fraud-detection-api:latest

    - name: Push Docker Image to Docker Hub
      run: |
        docker push ${{ secrets.DOCKER_USERNAME }}/fraud-detection-api:latest

    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG }}" | base64 --decode > kubeconfig.yaml
        export KUBECONFIG=kubeconfig.yaml

    - name: Deploy to Kubernetes
      run: |
        kubectl apply -f fraud-detection-deployment.yml
        kubectl apply -f fraud-detection-service.yml

    {
  "clientId": "your-client-id",
  "clientSecret": "your-client-secret",
  "subscriptionId": "your-subscription-id",
  "tenantId": "your-tenant-id",
  "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",
  "resourceManagerEndpointUrl": "https://management.azure.com/",
  "activeDirectoryGraphResourceId": "https://graph.windows.net/",
  "sqlManagementEndpointUrl": "https://management.core.windows.net/",
  "galleryEndpointUrl": "https://gallery.azure.com/",
  "managementEndpointUrl": "https://management.core.windows.net/"
}

```

---

## **🔹 Step 6: Set Up GitHub Secrets**
Store these in **GitHub → Repo → Settings → Secrets**  
✅ `DOCKER_USERNAME` → Your Docker Hub username  
✅ `DOCKER_PASSWORD` → Your Docker Hub password  
✅ `KUBE_CONFIG` → Base64 encoded `kubectl config` file for access to your K8s cluster  

---

## **🔹 Step 7: Deployment Workflow (Automatic)**
📌 **When you push code (`git push`) to `main` branch:**  
🔹 GitHub Actions automatically triggers CI/CD.  
🔹 Docker image is built and pushed to Docker Hub.  
🔹 Image is deployed to Kubernetes.  
🔹 Fraud API is accessible at your **Kubernetes LoadBalancer URL**.  

---

## **🔹 Step 8: Monitor & Scale**
**Monitor the Deployment:**  
```bash
kubectl get pods
kubectl get services
```
📌 **To View Logs**  
```bash
kubectl logs -l app=fraud-detection
```
📌 **To Scale** (Example: Scale to 5 replicas)  
```bash
kubectl scale deployment fraud-detection-api --replicas=5
```

---

### **🚀 Deployment Summary**
| **Step** | **Action** | **Tool Used** |
|--------|-------------|-------------|
| Step 1 | Train ML Model & Save | Python, Scikit-learn |
| Step 2 | Expose API | Flask |
| Step 3 | Containerize API | Docker |
| Step 4 | Automate Deployment | GitHub Actions |
| Step 5 | Push Docker Image | Docker Hub |
| Step 6 | Deploy with K8s | Kubernetes |
| Step 7 | Monitor & Scale | `kubectl`, Prometheus |

---
🎯 **Final Output: A real-time fraud detection API** accessible from **Kubernetes Load Balancer**  
📌 Ready for **real-world production ML deployment!** 🚀  

---

Let me know if you need improvements or integrations like **Prometheus for monitoring!** 🚀