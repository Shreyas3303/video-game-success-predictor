import requests
import os
import pandas as pd
from dotenv import load_dotenv
import time
import csv

# Load environment variables from .env file
load_dotenv()

class IGDBAPI:
    def __init__(self):
        self.client_id = os.getenv('IGDB_CLIENT_ID')
        self.access_token = os.getenv('IGDB_ACCESS_TOKEN')
        self.base_url = 'https://api.igdb.com/v4'
        
        if not self.client_id or not self.access_token:
            raise ValueError("Please set IGDB_CLIENT_ID and IGDB_ACCESS_TOKEN in your .env file")

    def get_games(self, offset=0, limit=500, min_rating=5):
        headers = {
            'Client-ID': self.client_id,
            'Authorization': f'Bearer {self.access_token}'
        }
        
        # Define the fields we want to get
        fields = [
            'checksum', 'slug', 'rating', 'rating_count', 'aggregated_rating',
            'aggregated_rating_count', 'total_rating', 'total_rating_count',
            'follows', 'hypes', 'game_type', 'game_status',
            'first_release_date', 'genres', 'platforms', 'franchises',
            'game_engines', 'game_modes', 'similar_games',
            'summary', 'player_perspectives'
        ]
        
        # Build the query with filters
        query = f'''
            fields {",".join(fields)};
            where (aggregated_rating >= {min_rating} | rating >= {min_rating}) 
            & first_release_date != null 
            & first_release_date >= {int(time.time()) - (25 * 365 * 24 * 60 * 60)};
            limit {limit};
            offset {offset};
        '''
        
        response = requests.post(
            f'{self.base_url}/games',
            headers=headers,
            data=query
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API request failed with status code {response.status_code}: {response.text}")

    def get_all_games(self, min_rating=70):
        all_games = []
        offset = 0
        limit = 500  # Maximum allowed by IGDB API
        
        while True:
            print(f"Fetching games {offset} to {offset + limit}...")
            games = self.get_games(offset=offset, limit=limit, min_rating=min_rating)
            
            if not games:  # No more games to fetch
                break
                
            all_games.extend(games)
            offset += limit
            
            # Sleep to respect rate limits
            time.sleep(0.25)
            
            # Optional: Break after certain number of games for testing
            if len(all_games) >= 30000:
                break
        
        return all_games

def save_to_csv(games, filename='igdb_games.csv'):
    # Convert to DataFrame
    df = pd.DataFrame(games)
    
    # Save to CSV with proper escaping and quoting
    df.to_csv(filename, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
    print(f"Saved {len(games)} games to {filename}")

if __name__ == "__main__":
    try:
        api = IGDBAPI()
        min_rating = 5  # You can adjust this value
        print(f"Fetching games with minimum rating of {min_rating}...")
        games = api.get_all_games(min_rating=min_rating)
        print(f"Retrieved {len(games)} games")
        save_to_csv(games)
    except Exception as e:
        print(f"Error: {e}") 