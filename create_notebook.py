import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create markdown cells
markdown_cells = [
    """# Video Game Success Predictor

This notebook analyzes video game metadata to predict total ratings using a Random Forest model.""",
    
    "## Import Required Libraries",
    
    "## Load and Preprocess Data",
    
    "## Feature Engineering",
    
    "## Handle Multi-label Features",
    
    "## Handle Missing Values",
    
    "## Prepare Features and Target",
    
    "## Train-Test Split and Model Training",
    
    "## Model Evaluation",
    
    "## Feature Importance Analysis"
]

# Create code cells
code_cells = [
    """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import ast""",

    """# Load data
df = pd.read_csv("games_expanded.csv")

# Drop irrelevant or high-cardinality columns
df.drop(columns=["id", "slug", "summary", "checksum", "game_type"], inplace=True)""",

    """# Convert release date
df['release_date'] = pd.to_datetime(df['first_release_date'], unit='s')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_dayofweek'] = df['release_date'].dt.dayofweek
df.drop(columns=['first_release_date', 'release_date'], inplace=True)

# Drop rows with missing target
df = df.dropna(subset=['total_rating'])""",

    """# Helper function to parse multi-label text fields
def parse_multilabel(col):
    return df[col].dropna().apply(lambda x: x.split(', '))

# Multi-label binarization
mlb = MultiLabelBinarizer()
multilabel_fields = ['genres', 'platforms', 'game_modes', 'player_perspectives']

for field in multilabel_fields:
    df[field] = df[field].fillna('')
    df[field] = df[field].apply(lambda x: x.split(', ') if x else [])
    encoded = mlb.fit_transform(df[field])
    encoded_df = pd.DataFrame(encoded, columns=[f"{field}_{cls}" for cls in mlb.classes_])
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=[field], inplace=True)""",

    """# Fill missing numeric values with 0 or mean
df["rating"].fillna(df["rating"].mean(), inplace=True)
df["rating_count"].fillna(0, inplace=True)
df["aggregated_rating"].fillna(df["aggregated_rating"].mean(), inplace=True)
df["aggregated_rating_count"].fillna(0, inplace=True)
df["hypes"].fillna(0, inplace=True)
df["game_status"].fillna(0, inplace=True)

# Drop columns with too sparse data
df.drop(columns=["similar_games", "franchises", "game_engines"], inplace=True)""",

    """# Final feature set
X = df.drop(columns=["total_rating"])
y = df["total_rating"]

# Normalize numerical features
numerical_cols = X.select_dtypes(include=[np.number]).columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])""",

    """# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)""",

    """# Predict and evaluate
y_pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R^2:", r2_score(y_test, y_pred))""",

    """# Feature importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)[-20:]
plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.show()"""
]

# Create cells list
cells = []
for md, code in zip(markdown_cells, code_cells):
    cells.append(nbf.v4.new_markdown_cell(md))
    cells.append(nbf.v4.new_code_cell(code))

# Add cells to notebook
nb['cells'] = cells

# Add metadata
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {
            'name': 'ipython',
            'version': 3
        },
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.8.0'
    }
}

# Write the notebook to a file
with open('alternate_modelling.ipynb', 'w') as f:
    nbf.write(nb, f) 