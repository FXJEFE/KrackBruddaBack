import json
import os
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Path to the config file
CONFIG_PATH = 'C:\\Users\\Administrator\\Documents\\FXJEFE_Project\\config.json'

# Load the config file safely
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
except FileNotFoundError:
    print(f"Error: Could not find config file at {CONFIG_PATH}")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Config file has invalid format - {e}")
    exit(1)

# Set up logging
log_file = os.path.join(config['log_path'], 'train_models.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("Script started and configuration loaded successfully")

def train_model(data, features, target='label'):
    try:
        X = data[features]
        y = data[target]
        
        logging.info(f"Training with {len(features)} features: {features}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        score = model.score(X_test, y_test)
        logging.info(f"Model trained with test score: {score}")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        return None

def main():
    data_path = os.path.join(config['data_output_path'], 'training_data.csv')
    model_path = os.path.join(config['models_path'], 'my_model.pkl')
    
    try:
        data = pd.read_csv(data_path, encoding='utf-8')
        model = train_model(data, config['features'])
        if model:
            joblib.dump(model, model_path)
            logging.info(f"Model saved to {model_path}")
        else:
            logging.error("Model training failed, no model saved")
            exit(1)
    except Exception as e:
        logging.error(f"Error processing data: {e}")
        exit(1)

if __name__ == "__main__":
    main()