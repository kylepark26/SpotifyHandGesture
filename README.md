# BMWSpotifyIntegration

Hand-gesture Spotify controller using **OpenCV**, **MediaPipe Hands**, and **Spotify Web API**.  
Control playback with your webcam:
- **Open palm (1.5s)** → Pause/Play
- **Hold two fingers up (1.5s)** → Volume UP
- **Hold two fingers down (1.5s)** → Volume DOWN
- **Hold two fingers right (1.5s)** → Next Track
- **Hold two fingers left (1.5s)** → Previous Track
- Overlay shows album art, title/artist, and a progress bar

## Features
- Robust gestures tolerant to low FPS cameras
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
