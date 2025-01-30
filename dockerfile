# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy necessary files
COPY requirements.txt .
COPY app.py .
COPY churn_model.joblib .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the application port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]