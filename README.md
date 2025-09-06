# BMWSpotifyIntegration

Hand-gesture Spotify controller using **OpenCV**, **MediaPipe Hands**, and **Spotipy**.  
Control playback with your webcam:
- **Open palm (2s)** → Pause/Play
- **Hold index fingertip at left edge (0.5s)** → Previous track
- **Hold index fingertip at right edge (0.5s)** → Next track
- Overlay shows album art, title/artist, and a progress bar

https://github.com/kylepark26/BMWSpotifyIntegration

## Features
- Robust edge-dwell gestures tolerant to low FPS cameras
- Compact Spotify-style overlay (album cover, metadata, progress dot)
- macOS-friendly camera selection (avoids Continuity Camera by default)

## Requirements
- Python 3.9+ (tested on macOS/Apple Silicon)
- A webcam
- Spotify Premium (required for remote playback control)
- A Spotify app (Client ID/Secret) from https://developer.spotify.com

## Setup

```bash
# clone
git clone https://github.com/kylepark26/BMWSpotifyIntegration.git
cd BMWSpotifyIntegration

# create & activate a venv (name can vary)
python3 -m venv env
source env/bin/activate

# install deps
pip install -r requirements.txt
