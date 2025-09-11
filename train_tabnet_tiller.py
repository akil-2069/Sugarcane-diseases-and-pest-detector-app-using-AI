import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Generate synthetic dataset similar to your sample structure
def generate_synthetic_data(num_samples=100):
    questions = [
        "Are the stems (tillers) shorter than usual?",
        "Do you smell any unusual or rotten odor near the plant base?",
        "Can you easily pull the roots out of the soil?",
        "Are you noticing yellowing on the lower stems?",
        "Do the tillers appear dry or brittle?",
        "Is there a soft mushy texture near the base of the stem?",
        "Are any of the tillers wilted or drooping?",
        "Do you see any black or brown spots on the stem?",
        "Are ants or other insects more active around the tiller base?",
        "Has the tiller growth slowed down recently?"
    ]
    
    data = []
    for sample_id in range(1, num_samples + 1):
        sample_label = np.random.choice([0, 1])
        for question in questions:
            if sample_label == 1:
                answer = np.random.choice([0,1], p=[0.3, 0.7])
            else:
                answer = np.random.choice([0,1], p=[0.7, 0.3])
            data.append([sample_id, question, answer, sample_label])
    
    df = pd.DataFrame(data, columns=["Sample_ID", "Question", "Answer", "Label"])
    return df

# 2. Prepare data for TabNet training
def prepare_data(df):
    df_pivot = df.pivot(index="Sample_ID", columns="Question", values="Answer").reset_index()
    labels = df.groupby("Sample_ID")["Label"].first().reset_index()
    df_final = df_pivot.merge(labels, on="Sample_ID")
    X = df_final.drop(columns=["Sample_ID", "Label"])
    y = df_final["Label"]
    return X, y

def train_tabnet_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    
    clf = TabNetClassifier(seed=42, verbose=0)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=50,
        patience=10,
        batch_size=32,
        virtual_batch_size=16,
        num_workers=0,
        drop_last=False
    )
    
    preds = clf.predict(X_test)
    print(classification_report(y_test, preds))
    
    return clf

if __name__ == "__main__":
    # Generate synthetic data and save CSV
    df = generate_synthetic_data(num_samples=200)
    csv_path = "synthetic_tiller_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic data CSV to {csv_path}")
    
    # Prepare features and labels
    X, y = prepare_data(df)
    
    # Train the model
    model = train_tabnet_model(X, y)
    
    # Save the model
    model_path = "tabnet_model.zip"
    model.save_model(model_path)
    print(f"Saved TabNet model to {model_path}")
    