import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import skew, kurtosis
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .config import CNN_INPUT_LENGTH, MODELS_DIR

def extract_features(flux):
    """Extracts statistical features from a phase-folded light curve for Random Forest."""
    if len(flux) == 0:
        return np.zeros(10)
        
    depth = 1.0 - np.min(flux)
    std_out = np.std(flux[np.abs(np.linspace(-0.5, 0.5, len(flux))) > 0.1])
    snr = depth / std_out if std_out > 0 else 0
    
    # Simple duration proxy: fraction of points below 1-sigma
    duration_frac = np.sum(flux < (1.0 - std_out)) / len(flux)
    
    # Skewness and kurtosis
    skew_val = skew(flux)
    kurt_val = kurtosis(flux)
    
    # RMS
    rms_in = np.sqrt(np.mean(flux[np.abs(np.linspace(-0.5, 0.5, len(flux))) < 0.1]**2))
    rms_out = np.sqrt(np.mean(flux[np.abs(np.linspace(-0.5, 0.5, len(flux))) > 0.1]**2))
    
    # Dummy values for the rest to keep it simple for this demo
    symmetry = 1.0
    odd_even = 1.0
    sec_eclipse = 0.0
    
    return np.array([
        depth, snr, duration_frac, symmetry, odd_even, 
        sec_eclipse, skew_val, kurt_val, rms_in, rms_out
    ])

def generate_synthetic_data(base_flux, n_augment=40):
    """Generates synthetic positive and negative samples for ML training."""
    positives = []
    negatives = []
    
    # Add base flux as positive
    positives.append(base_flux)
    
    # Augment positives with noise
    for _ in range(n_augment):
        noise = np.random.normal(0, np.std(base_flux) * 0.5, len(base_flux))
        positives.append(base_flux + noise)
        
    # Generate negatives
    # 1. Scrambled
    for _ in range(n_augment // 4 + 1):
        scrambled = np.random.permutation(base_flux)
        negatives.append(scrambled)
        
    # 2. Pure noise
    for _ in range(n_augment // 4 + 1):
        noise = np.random.normal(1.0, np.std(base_flux), len(base_flux))
        negatives.append(noise)
        
    # 3. Sinusoidal (binary mimic)
    for _ in range(n_augment // 4 + 1):
        x = np.linspace(0, 4*np.pi, len(base_flux))
        sine = 1.0 - (np.max(1.0 - base_flux) * np.sin(x)**2)
        negatives.append(sine)
        
    # 4. Inverted
    for _ in range(n_augment // 4 + 1):
        inverted = 1.0 + (1.0 - base_flux)
        negatives.append(inverted)
        
    return positives, negatives

def prepare_dataset(filtered_dict, targets):
    """Builds the ML dataset from all available targets."""
    X_rf = []
    X_cnn = []
    y = []
    
    for target in targets:
        name = target['name']
        lc = filtered_dict.get(name)
        if lc is None:
            continue
            
        period = target.get('known_period', 1.0)
        folded = lc.fold(period=period)
        binned = folded.bin(bins=CNN_INPUT_LENGTH).remove_nans()
        
        flux = binned.flux.value
        
        # Ensure exact length
        if len(flux) > CNN_INPUT_LENGTH:
            flux = flux[:CNN_INPUT_LENGTH]
        elif len(flux) < CNN_INPUT_LENGTH:
            flux = np.pad(flux, (0, CNN_INPUT_LENGTH - len(flux)), 'edge')
            
        positives, negatives = generate_synthetic_data(flux)
        
        for p in positives:
            # PyTorch expects shape (channels, length) for Conv1d, so (1, 201)
            X_cnn.append(np.expand_dims(p, axis=0))
            X_rf.append(extract_features(p))
            y.append(1)
            
        for n in negatives:
            X_cnn.append(np.expand_dims(n, axis=0))
            X_rf.append(extract_features(n))
            y.append(0)
            
    return np.array(X_rf), np.array(X_cnn, dtype=np.float32), np.array(y, dtype=np.float32)

def train_random_forest(X, y):
    """Trains and evaluates the Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    metrics = {
        'rf_accuracy': accuracy_score(y_test, y_pred),
        'rf_precision': precision_score(y_test, y_pred),
        'rf_recall': recall_score(y_test, y_pred),
        'rf_f1': f1_score(y_test, y_pred)
    }
    
    print("Random Forest Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    return rf, metrics

class ExoplanetCNN(nn.Module):
    def __init__(self, input_length):
        super(ExoplanetCNN, self).__init__()
        # Conv1d expects input of shape (batch, channels, length)
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate size after pooling
        pooled_length = input_length // 2 // 2
        self.fc1 = nn.Linear(32 * pooled_length, 64)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

def train_cnn(X, y):
    """Trains a simple 1D CNN using PyTorch."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).view(-1, 1))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = ExoplanetCNN(input_length=CNN_INPUT_LENGTH)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training PyTorch CNN...")
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    acc = correct / total
    print("CNN Results:")
    print(f"  cnn_accuracy: {acc:.4f}")
    
    torch.save(model.state_dict(), MODELS_DIR / "cnn_model.pth")
    return model, {'cnn_accuracy': acc}

def run_ml_pipeline(filtered_dict, targets, skip_cnn=False):
    """Main entry point for ML classification."""
    print("Preparing dataset for ML...")
    X_rf, X_cnn, y = prepare_dataset(filtered_dict, targets)
    
    if len(y) == 0:
        print("Not enough data to train ML models.")
        return [{} for _ in targets]
        
    rf_model, rf_metrics = train_random_forest(X_rf, y)
    
    cnn_metrics = {}
    if not skip_cnn:
        cnn_model, cnn_metrics = train_cnn(X_cnn, y)
        
    metrics = {**rf_metrics, **cnn_metrics}
    
    return [metrics for _ in targets]
