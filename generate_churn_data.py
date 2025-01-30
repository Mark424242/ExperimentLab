import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
data = {
    'customerID': [f'C{i:04d}' for i in range(1, n_samples + 1)],
    'gender': np.random.choice(['Male', 'Female'], size=n_samples),
    'SeniorCitizen': np.random.choice([0, 1], size=n_samples),
    'Partner': np.random.choice(['Yes', 'No'], size=n_samples),
    'Dependents': np.random.choice(['Yes', 'No'], size=n_samples),
    'tenure': np.random.randint(1, 72, size=n_samples),  # Tenure in months
    'PhoneService': np.random.choice(['Yes', 'No'], size=n_samples),
    'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], size=n_samples),
    'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples),
    'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples),
    'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples),
    'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples),
    'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples),
    'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples),
    'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples),
    'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples),
    'PaperlessBilling': np.random.choice(['Yes', 'No'], size=n_samples),
    'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], size=n_samples),
    'MonthlyCharges': np.round(np.random.uniform(20, 120, size=n_samples), 2),
    'TotalCharges': np.round(np.random.uniform(100, 8000, size=n_samples), 2),
    'Churn': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])  # 30% churn rate
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('customer_churn.csv', index=False)
print("customer_churn.csv file created successfully!")