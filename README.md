# Video Game Success Predictor

A machine learning project that predicts video game success using data from IGDB (Internet Game Database).

## Project Structure

```
video-game-success-predictor/
├── data/               # Data files (CSV)
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── src/              # Source code
├── config/           # Configuration files
└── scripts/          # Utility scripts
```

## Setup

1. Create a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up IGDB API credentials:
   - Create a `.env` file in the root directory
   - Add your IGDB API credentials:
     ```
     IGDB_CLIENT_ID=your_client_id
     IGDB_CLIENT_SECRET=your_client_secret
     ```

## Data

The project uses data from IGDB. The main data files are:
- `data/igdb_games.csv`: Raw game data from IGDB
- `data/games_expanded.csv`: Processed and expanded game data

## Usage

1. Data Collection:
```bash
python scripts/fetch_additional_data.py
```

2. Model Training:
```bash
python src/predict_game_rating.py
```

3. Exploration and Analysis:
   - Open `notebooks/exploration.ipynb` for data exploration
   - Open `notebooks/modelling.ipynb` for model development

## Model

The project uses a machine learning model to predict video game success based on various features. The trained model is saved in `models/best_model.joblib`.