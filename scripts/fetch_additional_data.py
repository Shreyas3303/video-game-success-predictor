import requests
import os
import pandas as pd
from dotenv import load_dotenv
import time
import json
import csv

# Load environment variables from .env file
load_dotenv()

class IGDBAdditionalData:
    def __init__(self):
        self.client_id = os.getenv('IGDB_CLIENT_ID')
        self.access_token = os.getenv('IGDB_ACCESS_TOKEN')
        self.base_url = 'https://api.igdb.com/v4'
        
        if not self.client_id or not self.access_token:
            raise ValueError("Please set IGDB_CLIENT_ID and IGDB_ACCESS_TOKEN in your .env file")

    def get_reference_data(self, endpoint='endpoint_name', fields=None):
        """
        Fetch all data from a reference endpoint (e.g., genres, platforms)
        
        Args:
            endpoint (str): The IGDB API endpoint to query
            fields (list): List of fields to fetch from the endpoint
        """
        headers = {
            'Client-ID': self.client_id,
            'Authorization': f'Bearer {self.access_token}'
        }
        
        # Build the query to get all data from the endpoint
        query = f'''
            fields {",".join(fields)};
            limit 500;
        '''
        
        all_data = []
        offset = 0
        
        while True:
            # Add offset to query
            query_with_offset = query + f' offset {offset};'
            
            response = requests.post(
                f'{self.base_url}/{endpoint}',
                headers=headers,
                data=query_with_offset
            )
            
            if response.status_code == 200:
                batch_data = response.json()
                if not batch_data:  # No more data
                    break
                    
                all_data.extend(batch_data)
                print(f"Fetched {len(batch_data)} {endpoint}")
                offset += 500
            else:
                print(f"Error fetching {endpoint}: {response.status_code}")
                print(response.text)
                break
            
            time.sleep(0.25)
        
        return all_data

    def expand_reference_fields(self, csv_file='igdb_games.csv', reference_fields=None):
        """
        Replace reference IDs with their names in the games CSV.
        
        Args:
            csv_file (str): Path to the games CSV file
            reference_fields (dict): Dictionary mapping field names to their endpoint and fields to fetch
        """
        # Read the games CSV
        try:
            df = pd.read_csv(csv_file, 
                           quoting=csv.QUOTE_ALL, 
                           escapechar='\\',
                           on_bad_lines='warn',  # Warn about problematic lines instead of failing
                           encoding='utf-8')
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            print("Attempting to read with different parameters...")
            # Try reading with more lenient parameters
            df = pd.read_csv(csv_file,
                           quoting=csv.QUOTE_MINIMAL,  # More lenient quoting
                           escapechar='\\',
                           on_bad_lines='warn',
                           encoding='utf-8')
        
        print(f"Successfully read CSV with {len(df.columns)} columns")
        print("Column names:", df.columns.tolist())
        
        # For each reference field, fetch the data and create a mapping
        for field, config in reference_fields.items():
            print(f"\nProcessing {field}...")
            
            # Fetch all data for this endpoint
            reference_data = self.get_reference_data(
                endpoint=config['endpoint'],
                fields=config['fields']
            )
            
            # Create a mapping of IDs to names
            # Handle different field structures for different endpoints
            reference_map = {item['id']: item['slug'] for item in reference_data}
            
            print(f"Created reference map with {len(reference_map)} entries")
            print("Sample of reference map:", dict(list(reference_map.items())[:3]))
            
            # Function to convert IDs to names
            def convert_ids_to_names(ids_str):
                if pd.isna(ids_str):
                    return None
                try:
                    # Try to parse as JSON array
                    ids = json.loads(ids_str)
                    if isinstance(ids, list):
                        # Convert each ID to its name and join with commas
                        names = [reference_map.get(id, '') for id in ids]
                        print(f"Converting IDs {ids} to names {names}")  # Debug print
                        return ', '.join(filter(None, names))
                except:
                    # If not an array, try as single ID
                    try:
                        result = reference_map.get(int(ids_str), '')
                        print(f"Converting single ID {ids_str} to {result}")  # Debug print
                        return result
                    except:
                        return ids_str
                return ids_str
            
            # Replace the IDs with names in the original column
            df[field] = df[field].apply(convert_ids_to_names)
            print(f"Updated {field} column with names")
        
        # Save the updated games CSV
        output_file = 'games_expanded.csv'
        df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL, escapechar='\\', doublequote=True)
        print(f"\nSaved updated games data to {output_file}")
        return df

def main():
    try:
        api = IGDBAdditionalData()
        
        # Define the reference fields to expand
        reference_fields = {
            'genres': {
                'endpoint': 'genres',
                'fields': ['id', 'slug']
            },
            'platforms': {
                'endpoint': 'platforms',
                'fields': ['id', 'slug']
            },
            'game_engines': {
                'endpoint': 'game_engines',
                'fields': ['id', 'slug']
            },
            'game_modes': {
                'endpoint': 'game_modes',
                'fields': ['id', 'slug']
            },
            'player_perspectives': {
                'endpoint': 'player_perspectives',
                'fields': ['id', 'slug']
            },
            'similar_games': {
                'endpoint': 'games',
                'fields': ['id', 'slug']
            },
            'franchises': {
                'endpoint': 'franchises',
                'fields': ['id', 'slug']
            },
        }
        
        # Expand the reference fields in the games CSV
        api.expand_reference_fields(
            csv_file='igdb_games.csv',
            reference_fields=reference_fields
        )
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 