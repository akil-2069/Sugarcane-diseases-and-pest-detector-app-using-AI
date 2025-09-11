import pandas as pd
import json, os
# Update this path if your training CSV is elsewhere:
train_csv = os.path.join('data', yellow_leaf_synthetic_data.csv)

if not os.path.exists(train_csv):
    raise SystemExit(f"Training CSV not found: {train_csv}")

df = pd.read_csv(train_csv)

# Try to detect target column name
candidates = ['Label', 'label', 'Target', 'target', 'Outcome', 'outcome']
target_col = None
for c in candidates:
    if c in df.columns:
        target_col = c
        break
if target_col is None:
    # assume last column is label
    target_col = df.columns[-1]

features = [c for c in df.columns if c != target_col]
os.makedirs('tabnet_models', exist_ok=True)

feat_json_path = os.path.join('tabnet_models', 'tabnet_yellow_leaf_features.json')
with open(feat_json_path, 'w') as f:
    json.dump(features, f, indent=2)
print("Saved features list to:", feat_json_path)

# make a template one-row CSV (zeros)
template = pd.DataFrame([ {c: 0 for c in features} ])
template_csv_path = os.path.join('tabnet_models', 'tabnet_yellow_leaf_template.csv')
template.to_csv(template_csv_path, index=False)
print("Saved template CSV to:", template_csv_path)
print("Feature count:", len(features))
