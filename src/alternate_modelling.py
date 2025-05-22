# Predict total_rating from game metadata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("games_expanded.csv")

# Print initial NaN counts
print("\nInitial NaN counts:")
print(df.isna().sum())

# Drop irrelevant or high-cardinality columns
df.drop(columns=["id", "slug", "summary", "checksum", "game_type", "rating", "rating_count", 
                 "aggregated_rating", "aggregated_rating_count"], inplace=True)

# Convert release date and create more features
df['release_date'] = pd.to_datetime(df['first_release_date'], unit='s')
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_dayofweek'] = df['release_date'].dt.dayofweek
df['release_quarter'] = df['release_date'].dt.quarter
df['release_dayofyear'] = df['release_date'].dt.dayofyear
df['is_weekend'] = df['release_dayofweek'].isin([5, 6]).astype(int)
df.drop(columns=['first_release_date', 'release_date'], inplace=True)

# Handle outliers in total_rating using IQR method
Q1 = df['total_rating'].quantile(0.25)
Q3 = df['total_rating'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['total_rating'] >= lower_bound) & (df['total_rating'] <= upper_bound)]

# Drop rows with missing target
df = df.dropna(subset=['total_rating'])

# Multi-label binarization
multilabel_fields = ['genres', 'platforms', 'game_modes', 'player_perspectives']
mlb_dict = {}  # Dictionary to store MultiLabelBinarizers for each field

for field in multilabel_fields:
    df[field] = df[field].fillna('')
    df[field] = df[field].apply(lambda x: x.split(', ') if x else [])
    mlb = MultiLabelBinarizer()  # Create a new MultiLabelBinarizer for each field
    encoded = mlb.fit_transform(df[field])
    encoded_df = pd.DataFrame(encoded, columns=[f"{field}_{cls}" for cls in mlb.classes_])
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=[field], inplace=True)
    mlb_dict[field] = mlb  # Store the MultiLabelBinarizer

# Fill missing numeric values with 0
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df[col] = df[col].fillna(0)

# Drop columns with too sparse data
df.drop(columns=["similar_games", "franchises", "game_engines"], inplace=True)

# Print NaN counts after preprocessing
print("\nNaN counts after preprocessing:")
print(df.isna().sum())

# Final feature set
X = df.drop(columns=["total_rating"])
y = df["total_rating"]

# Print shape of final dataset
print("\nFinal dataset shape:", X.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to try
models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Ridge': Ridge()
}

# Define parameter grids for each model
param_grids = {
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    },
    'Gradient Boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0]
    }
}

# Results storage
results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('model', model)
    ])
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        {f'model__{k}': v for k, v in param_grids[name].items()},
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Best Parameters': grid_search.best_params_
    }
    
    print(f"{name} Results:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")

# Visualize results
plt.figure(figsize=(15, 5))

# Plot 1: Model Comparison
plt.subplot(1, 3, 1)
metrics = ['MAE', 'RMSE', 'R2']
x = np.arange(len(metrics))
width = 0.25

for i, (model_name, result) in enumerate(results.items()):
    values = [result[metric] for metric in metrics]
    plt.bar(x + i*width, values, width, label=model_name)

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Comparison')
plt.xticks(x + width, metrics)
plt.legend()

# Plot 2: Actual vs Predicted for best model
best_model_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
best_model = models[best_model_name]
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title(f'Actual vs Predicted ({best_model_name})')

# Plot 3: Feature Importance for Random Forest
if isinstance(best_model, RandomForestRegressor):
    plt.subplot(1, 3, 3)
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10 features
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Feature Importances')

plt.tight_layout()
plt.show()

# Print final summary
print("\nFinal Model Summary:")
print(f"Best performing model: {best_model_name}")
print(f"Best R2 Score: {results[best_model_name]['R2']:.4f}")
print(f"Best Parameters: {results[best_model_name]['Best Parameters']}")

# Save the best model and MultiLabelBinarizers
from predict_game_rating import save_model
save_model(best_model, mlb_dict, 'best_model.joblib', 'mlb_dict.joblib')
