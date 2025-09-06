import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Replace these with your own credentials
CLIENT_ID = 'aba0a76ca1a745828758cdd4d01e06a9'
CLIENT_SECRET = 'f2110e6458494e699d9688a7080b5cdb'
REDIRECT_URI = 'http://localhost:8888/callback'

# Initialize the Spotify client with updated scope
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,
                                               client_secret=CLIENT_SECRET,
                                               redirect_uri=REDIRECT_URI,
                                               scope="user-library-read user-modify-playback-state user-read-playback-state"))

# Step 1: Get the list of available devices
devices = sp.devices()
print("Available Devices:")
for device in devices['devices']:
    print(f"Device: {device['name']}, ID: {device['id']}, Type: {device['type']}")

# Step 2: Set the device ID for your phone (replace with your actual phone's device ID)
phone_device_id = '618646adc34094e98cf93c5d1f8d0799d791b7fb'  # Replace with the correct ID of your phone

# Step 3: Control playback on your phone

# To start or resume playback:
sp.start_playback(device_id=phone_device_id)

# To pause playback:
# sp.pause_playback(device_id=phone_device_id)

# Step 4: Test: Print current playback status
current_playback = sp.current_playback()
print("Current Playback Status:")
print(current_playback)