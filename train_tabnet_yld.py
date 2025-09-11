import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Create Yellow Leaf Disease synthetic dataset generator
def generate_yellow_leaf_data(num_samples=100):
    questions_answers = [
        ("Are the leaves turning yellow starting from the tips?", 1),
        ("Do you notice any brown streaks on the leaf veins?", 0),
        ("Is there a sticky residue on the leaf surface?", 1),
        ("Are the leaves curling inward or downward?", 0),
        ("Do you see any tiny white spots on the leaves?", 1),
        ("Is there a foul or unusual smell from the leaves?", 0),
        ("Are the leaf edges turning brown or dry?", 1),
        ("Do the leaves have patches of light green or pale color?", 0),
        ("Is the plant growth slower than usual?", 1),
        ("Are the leaves falling off prematurely?", 0),
        ("Do you see black fungal spots on the leaves?", 1),
        ("Are ants or other insects frequently found on the leaves?", 0),
        ("Is there any powdery white coating on the leaves?", 1),
        ("Do you see small holes or eaten parts on the leaves?", 0),
        ("Are the younger leaves affected more than older ones?", 1),
        ("Is there any sticky honeydew on the underside of leaves?", 0),
        ("Do the leaves feel brittle or dry to touch?", 1),
        ("Are there any web-like structures on the leaves?", 0),
        ("Is there any mold or mildew on the leaf surface?", 1),
        ("Are the leaves drooping or wilted during the day?", 0),
        ("Do you see any yellow rings or spots on the leaves?", 1),
        ("Are the leaves thinner or smaller than normal?", 0),
        ("Is there any rust-colored powder on the leaves?", 1),
        ("Do you notice any leaf veins turning red or purple?", 0),
        ("Are the leaves brittle and breaking easily?", 1),
        ("Is the plant showing signs of nutrient deficiency?", 0),
        ("Are there any whiteflies or aphids on the leaves?", 1),
        ("Do the leaves have translucent patches?", 0),
        ("Is there a sticky substance attracting ants?", 1),
        ("Are the leaves clustered tightly together unusually?", 0),
        ("Is the leaf surface rough or bumpy?", 1),
        ("Are the plants less productive than usual?", 0),
        ("Do you see any holes forming in the leaf blades?", 1),
        ("Are the edges of the leaves curling upward?", 0),
        ("Do you notice any faint yellow halos around spots?", 1),
        ("Is there excessive moisture or dew on the leaves?", 0),
        ("Are the leaves uneven in size and shape?", 1),
        ("Do you see any leaves that look scorched or sunburned?", 0),
        ("Are the leaves losing their natural shine?", 1),
        ("Is there a sticky film on the leaf surface?", 0),
        ("Do the affected leaves wilt faster during hot weather?", 1),
        ("Are any parts of the leaf transparent or thin?", 0),
        ("Do you notice any fungus-like growth near leaf stems?", 1),
        ("Are the leaves changing color faster than normal?", 0),
        ("Is the plant less vigorous than healthy plants?", 1),
        ("Do you observe any yellowing around leaf margins?", 0),
        ("Are the leaves curling unevenly or asymmetrically?", 1),
        ("Do you see any spots that look oily or greasy?", 0)
    ]
    
    data = []
    for sample_id in range(1, num_samples + 1):
        # Assign label 1 (diseased) or 0 (healthy) randomly
        label = np.random.choice([0, 1])
        
        for question, base_answer in questions_answers:
            # To add some variability:
            # if label==1 (diseased), answers tend to be like base_answer but with 80% chance
            # if label==0 (healthy), answers tend to be opposite with 80% chance
            
            if label == 1:
                answer = np.random.choice([base_answer, 1 - base_answer], p=[0.8, 0.2])
            else:
                answer = np.random.choice([1 - base_answer, base_answer], p=[0.8, 0.2])
            
            data.append([sample_id, question, answer, label])
    
    df = pd.DataFrame(data, columns=["Sample_ID", "Question", "Answer", "Label"])
    return df

# 2. Prepare data for TabNet training (same pivot method)
def prepare_data(df):
    df_pivot = df.pivot(index="Sample_ID", columns="Question", values="Answer").reset_index()
    labels = df.groupby("Sample_ID")["Label"].first().reset_index()
    df_final = df_pivot.merge(labels, on="Sample_ID")
    X = df_final.drop(columns=["Sample_ID", "Label"])
    y = df_final["Label"]
    return X, y

# 3. Train TabNet model (same as before)
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
    # Generate synthetic Yellow Leaf Disease data and save CSV
    df = generate_yellow_leaf_data(num_samples=200)
    csv_path = "yellow_leaf_synthetic_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved Yellow Leaf synthetic data CSV to {csv_path}")
    
    # Prepare data for training
    X, y = prepare_data(df)
    
    # Train the model
    model = train_tabnet_model(X, y)
    
    # Save the model
    model_path = "yellow_leaf_tabnet_model.zip"
    model.save_model(model_path)
    print(f"Saved TabNet model to {model_path}")
