# BMWSpotifyIntegration

Hand-gesture Spotify controller using **OpenCV**, **MediaPipe Hands**, and **Spotipy**.  
Control playback with your webcam:
- **Open palm (2s)** → Pause/Play
- **Hold index fingertip at left edge (0.5s)** → Previous track
- **Hold index fingertip at right edge (0.5s)** → Next track
- Overlay shows album art, title/artist, and a progress bar

## Features
- Robust edge-dwell gestures tolerant to low FPS cameras
- Compact Spotify-style overlay (album cover, metadata, progress dot)
- macOS-friendly camera selection (avoids Continuity Camera by default)

## Setup

```bash
# clone
git clone https://github.com/kylepark26/BMWSpotifyIntegration.git
cd BMWSpotifyIntegration

python3 -m venv env
source env/bin/activate

pip install -r requirements.txt
