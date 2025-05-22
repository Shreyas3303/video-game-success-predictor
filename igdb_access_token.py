import requests
import os
from typing import Dict, Optional

def get_twitch_token(client_id: str, client_secret: str) -> Optional[Dict]:
    """
    Get a Twitch OAuth2 token using client credentials.
    
    Args:
        client_id (str): Your Twitch Client ID
        client_secret (str): Your Twitch Client Secret
        
    Returns:
        Optional[Dict]: The response containing the access token if successful, None if failed
    """
    url = "https://id.twitch.tv/oauth2/token"
    
    params = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    
    try:
        response = requests.post(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting Twitch token: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    client_id = os.getenv("TWITCH_CLIENT_ID")
    client_secret = os.getenv("TWITCH_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        print("Please set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET environment variables")
    else:
        token_response = get_twitch_token(client_id, client_secret)
        if token_response:
            print("Successfully obtained token:")
            print(token_response)
