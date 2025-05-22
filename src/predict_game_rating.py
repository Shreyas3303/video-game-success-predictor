import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, RobustScaler
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# Define paths relative to project root
MODEL_PATH = os.path.join('models', 'best_model.joblib')
MLB_PATH = os.path.join('models', 'mlb_dict.joblib')

def preprocess_game_data(game_data, mlb_dict=None):
    """
    Preprocess a single game's data to match the training data format.
    
    Parameters:
    game_data (dict): Dictionary containing game information
    mlb_dict (dict): Dictionary of pre-fitted MultiLabelBinarizers for each field
    """
    # Convert to DataFrame
    df = pd.DataFrame([game_data])
    
    # Convert release date
    df['release_date'] = pd.to_datetime(df['first_release_date'], unit='s')
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_dayofweek'] = df['release_date'].dt.dayofweek
    df['release_quarter'] = df['release_date'].dt.quarter
    df['release_dayofyear'] = df['release_date'].dt.dayofyear
    df['is_weekend'] = df['release_dayofweek'].isin([5, 6]).astype(int)
    df.drop(columns=['first_release_date', 'release_date'], inplace=True)
    
    # Multi-label binarization
    multilabel_fields = ['genres', 'platforms', 'game_modes', 'player_perspectives']
    
    for field in multilabel_fields:
        df[field] = df[field].fillna('')
        df[field] = df[field].apply(lambda x: x.split(', ') if x else [])
        if mlb_dict and field in mlb_dict:
            encoded = mlb_dict[field].transform(df[field])
            encoded_df = pd.DataFrame(encoded, columns=[f"{field}_{cls}" for cls in mlb_dict[field].classes_])
        else:
            mlb = MultiLabelBinarizer()
            encoded = mlb.fit_transform(df[field])
            encoded_df = pd.DataFrame(encoded, columns=[f"{field}_{cls}" for cls in mlb.classes_])
        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=[field], inplace=True)
    
    # Fill missing numeric values with 0
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(0)
    
    # Ensure columns are in the exact same order as the trained model
    expected_features = [
        'total_rating_count', 'hypes', 'game_status',
        'release_year', 'release_month', 'release_dayofweek', 'release_quarter', 'release_dayofyear', 'is_weekend',
        'genres_adventure', 'genres_arcade', 'genres_card-and-board-game', 'genres_fighting', 'genres_hack-and-slash-beat-em-up',
        'genres_indie', 'genres_moba', 'genres_music', 'genres_pinball', 'genres_platform', 'genres_point-and-click',
        'genres_puzzle', 'genres_quiz-trivia', 'genres_racing', 'genres_real-time-strategy-rts', 'genres_role-playing-rpg',
        'genres_shooter', 'genres_simulator', 'genres_sport', 'genres_strategy', 'genres_tactical',
        'genres_turn-based-strategy-tbs', 'genres_visual-novel',
        'platforms_3ds', 'platforms_64dd', 'platforms_amazon-fire-tv', 'platforms_amiga', 'platforms_android',
        'platforms_arcade', 'platforms_arduboy', 'platforms_atari7800', 'platforms_blackberry', 'platforms_blu-ray-player',
        'platforms_browser', 'platforms_c64', 'platforms_daydream', 'platforms_dc', 'platforms_digiblast',
        'platforms_dos', 'platforms_dvd-player', 'platforms_evercade', 'platforms_fairchild-channel-f', 'platforms_famicom',
        'platforms_gba', 'platforms_gbc', 'platforms_gear-vr', 'platforms_genesis-slash-megadrive', 'platforms_gizmondo',
        'platforms_ios', 'platforms_leapster', 'platforms_leapster-explorer-slash-leadpad-explorer', 'platforms_linux',
        'platforms_mac', 'platforms_meta-quest-2', 'platforms_meta-quest-3', 'platforms_mobile', 'platforms_msx',
        'platforms_n64', 'platforms_nds', 'platforms_neo-geo-cd', 'platforms_neogeoaes', 'platforms_neogeomvs',
        'platforms_nes', 'platforms_new-nintendo-3ds', 'platforms_ngage', 'platforms_ngc', 'platforms_nintendo-dsi',
        'platforms_nuon', 'platforms_oculus-go', 'platforms_oculus-quest', 'platforms_oculus-rift', 'platforms_oculus-vr',
        'platforms_onlive-game-system', 'platforms_ooparts', 'platforms_ouya', 'platforms_palm-os', 'platforms_plug-and-play',
        'platforms_ps', 'platforms_ps2', 'platforms_ps3', 'platforms_ps4--1', 'platforms_ps5', 'platforms_psp',
        'platforms_psvita', 'platforms_psvr', 'platforms_psvr2', 'platforms_sega32', 'platforms_series-x-s',
        'platforms_sfam', 'platforms_snes', 'platforms_stadia', 'platforms_steam-vr', 'platforms_super-nes-cd-rom-system',
        'platforms_switch', 'platforms_switch-2', 'platforms_turbografx-16-slash-pc-engine-cd', 'platforms_visionos',
        'platforms_wii', 'platforms_wiiu', 'platforms_win', 'platforms_windows-mixed-reality', 'platforms_windows-mobile',
        'platforms_winphone', 'platforms_wonderswan', 'platforms_wonderswan-color', 'platforms_xbox', 'platforms_xbox360',
        'platforms_xboxone', 'platforms_zeebo', 'platforms_zod',
        'game_modes_battle-royale', 'game_modes_co-operative', 'game_modes_massively-multiplayer-online-mmo',
        'game_modes_multiplayer', 'game_modes_single-player', 'game_modes_split-screen',
        'player_perspectives_auditory', 'player_perspectives_bird-view-slash-isometric',
        'player_perspectives_first-person', 'player_perspectives_side-view', 'player_perspectives_text',
        'player_perspectives_third-person', 'player_perspectives_virtual-reality'
    ]
    
    # Add any missing columns with 0 values
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reorder columns to match expected order
    df = df[expected_features]
    
    return df

def predict_game_rating(game_data, model_path=MODEL_PATH, mlb_path=MLB_PATH):
    """
    Predict the rating for a new game.
    
    Parameters:
    game_data (dict): Dictionary containing game information
    model_path (str): Path to the saved model file
    mlb_path (str): Path to the saved MultiLabelBinarizer dictionary
    """
    try:
        # Load the model and MultiLabelBinarizers
        model = joblib.load(model_path)
        mlb_dict = joblib.load(mlb_path)
        
        # Preprocess the game data
        processed_data = preprocess_game_data(game_data, mlb_dict)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        
        return prediction
    
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {str(e)}")
        return None
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None

def save_model(model, mlb_dict, model_path=MODEL_PATH, mlb_path=MLB_PATH):
    """
    Save the trained model and MultiLabelBinarizers to files.
    
    Parameters:
    model: Trained model object
    mlb_dict (dict): Dictionary of fitted MultiLabelBinarizers
    model_path (str): Path to save the model
    mlb_path (str): Path to save the MultiLabelBinarizer dictionary
    """
    try:
        # Ensure models directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        joblib.dump(model, model_path)
        joblib.dump(mlb_dict, mlb_path)
        print(f"Model and MultiLabelBinarizers saved successfully")
    except Exception as e:
        print(f"Error saving files: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Example game data
    example_game = {
        'first_release_date': 788918400,  # 1995-01-01
        'genres': 'pinball, quiz-trivia, card-and-board-game',  # Completely different genres
        'platforms': 'browser, mobile, ios',  # Completely different platforms
        'game_modes': 'battle-royale, massively-multiplayer-online-mmo',  # Completely different modes
        'player_perspectives': 'text, virtual-reality',  # Completely different perspectives
        'hypes': 5000,  # Unchanged
        'game_status': 1,  # Unchanged
        'total_rating_count': 6  # Unchanged
    }
    
    # Make prediction
    predicted_rating = predict_game_rating(example_game)
    
    if predicted_rating is not None:
        print(f"\nPredicted rating: {predicted_rating:.2f}")
        print("\nNote: To use this script with your trained model:")
        print("1. Run your training script (alternate_modelling.py)")
        print("2. Save the model and MultiLabelBinarizers using save_model() function")
        print("3. Use predict_game_rating() with your game data") 