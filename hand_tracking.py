import cv2
import mediapipe as mp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import requests
import numpy as np

# --------------------------
# MediaPipe Hands
# --------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# --------------------------
# OpenCV Camera (prefer Mac cam, not Continuity) + make it lighter
# --------------------------
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    for idx in (0, 2, 3):
        cap.release()
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            break
    if not cap.isOpened():
        raise RuntimeError("Could not open a camera. Disable Continuity Camera or try other indices.")

# Lower res for higher FPS (best-effort)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # try 1920 for 1080p if your FPS is ok
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Make the display window resizable and set an initial big size
cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Tracking", 1280, 800)

# --------------------------
# Spotify client
# --------------------------
# (Consider env vars for security in a real project)
CLIENT_ID = 'aba0a76ca1a745828758cdd4d01e06a9'
CLIENT_SECRET = 'f2110e6458494e699d9688a7080b5cdb'
REDIRECT_URI = 'http://localhost:8888/callback'

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope="user-library-read user-modify-playback-state user-read-playback-state"
))

def prefer_desktop_device_once():
    try:
        devices = sp.devices().get('devices', [])
        desktop = next((d for d in devices if d.get('type') == 'Computer'), None)
        if desktop:
            sp.transfer_playback(device_id=desktop['id'], force_play=False)
    except Exception:
        pass

prefer_desktop_device_once()

# --------------------------
# Gestures: open-palm pause/play + dwell-at-edges for prev/next
# --------------------------
music_paused = False
palm_open_time = 0.0            # start time when palm is detected

DWELL_REQUIRED = 0.50           # seconds to hold at edge to trigger
ACTION_COOLDOWN = 1.00          # seconds between prev/next actions
last_action_time = 0.0

EDGE_FRAC = 0.18                # left/right edge width as fraction of frame width
current_zone = None             # 'L', 'R', or None
zone_entry_time = 0.0

SHOW_ZONES = False              # set True to visualize the dwell zones

def is_palm_open(hand_landmarks):
    """Simple open-palm heuristic based on wrist-to-fingertip distances."""
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    def d(a, b):
        return np.hypot(a.x - b.x, a.y - b.y)

    threshold = 0.20
    return all([
        d(wrist, thumb_tip) > threshold,
        d(wrist, index_tip) > threshold,
        d(wrist, middle_tip) > threshold,
        d(wrist, ring_tip) > threshold,
        d(wrist, pinky_tip) > threshold
    ])

def which_zone(x_px, w):
    """Return 'L' if in left edge, 'R' if in right edge, else None."""
    left_limit = int(w * EDGE_FRAC)
    right_limit = int(w * (1.0 - EDGE_FRAC))
    if x_px <= left_limit:
        return 'L'
    if x_px >= right_limit:
        return 'R'
    return None

# --------------------------
# Spotify helpers
# --------------------------
def get_current_song_info():
    """Return (title, artist, album_cover_bgr, progress_ms, duration_ms)."""
    current = sp.current_playback()
    if current and current.get('item'):
        track = current['item']
        title = track['name']
        artist = ', '.join([a['name'] for a in track['artists']])
        duration_ms = track['duration_ms']
        progress_ms = current.get('progress_ms', 0)

        # Album cover
        cover = None
        try:
            album_cover_url = track['album']['images'][0]['url']
            response = requests.get(album_cover_url, timeout=2.0)
            arr = np.asarray(bytearray(response.content), dtype=np.uint8)
            cover = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
        except Exception:
            pass

        return title, artist, cover, progress_ms, duration_ms
    return None, None, None, None, None

# --------------------------
# UI drawing (top-right panel)
# --------------------------
def draw_top_left_panel(frame, title, artist, album_cover, progress_ms, duration_ms):
    """
    Draw a compact Spotify-like panel in the top-left:
    - semi-transparent background
    - 140x140 album cover
    - title + artist
    - progress bar
    """
    h, w = frame.shape[:2]
    panel_w, panel_h = 360, 170
    x0 = 12                 # <-- left margin
    y0 = 12                 # <-- top margin
    x1 = x0 + panel_w
    y1 = y0 + panel_h

    # translucent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), thickness=-1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    # album cover
    cover_size = 140
    if album_cover is not None:
        ac = cv2.resize(album_cover, (cover_size, cover_size))
        frame[y0+10:y0+10+cover_size, x0+10:x0+10+cover_size] = ac

    # text
    text_x = x0 + 10 + cover_size + 12
    text_y = y0 + 35
    if title:
        cv2.putText(frame, title[:28], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        text_y += 28
    if artist:
        cv2.putText(frame, artist[:32], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)

    # progress bar
    bar_x0 = x0 + 10
    bar_x1 = x1 - 10
    bar_y  = y0 + panel_h - 20
    cv2.line(frame, (bar_x0, bar_y), (bar_x1, bar_y), (160,160,160), 3, cv2.LINE_AA)

    if progress_ms is not None and duration_ms:
        frac = max(0.0, min(1.0, progress_ms / float(duration_ms)))
        dot_x = int(bar_x0 + frac * (bar_x1 - bar_x0))
        cv2.circle(frame, (dot_x, bar_y), 6, (255,255,255), -1, cv2.LINE_AA)

        # timestamps (mm:ss)
        def mmss(ms):
            s = int(ms/1000)
            return f"{s//60}:{s%60:02d}"
        cv2.putText(frame, mmss(progress_ms), (bar_x0, bar_y-8), cv2.FONT_HERSHEY_PLAIN, 1.0, (220,220,220), 1, cv2.LINE_AA)
        cv2.putText(frame, mmss(duration_ms), (bar_x1-40, bar_y-8), cv2.FONT_HERSHEY_PLAIN, 1.0, (220,220,220), 1, cv2.LINE_AA)

def draw_edge_guides(frame):
    """Optional: visualize left/right dwell zones."""
    if not SHOW_ZONES:
        return
    h, w = frame.shape[:2]
    left_limit = int(w * EDGE_FRAC)
    right_limit = int(w * (1.0 - EDGE_FRAC))
    cv2.rectangle(frame, (0, 0), (left_limit, h), (0, 255, 0), 2)
    cv2.rectangle(frame, (right_limit, 0), (w-1, h), (0, 255, 0), 2)

# --------------------------
# Main loop
# --------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror for natural interaction
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w = frame.shape[:2]
    draw_edge_guides(frame)

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            # ----- Pause/Play: open palm for 2s -----
            if is_palm_open(lm):
                if palm_open_time == 0.0:
                    palm_open_time = time.time()
                elif time.time() - palm_open_time >= 2.0:
                    current = sp.current_playback()
                    if current:
                        is_playing = current.get('is_playing')
                        if is_playing:
                            if not music_paused:
                                print("Palm 2s → Pause")
                                sp.pause_playback()
                                music_paused = True
                        else:
                            if music_paused:
                                print("Palm 2s → Play")
                                sp.start_playback()
                                music_paused = False
                    palm_open_time = 0.0
            else:
                palm_open_time = 0.0

            # ----- Prev/Next: dwell at left/right edge for 0.5s -----
            idx_tip = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x_px = int(idx_tip.x * w)
            zone = which_zone(x_px, w)
            now = time.time()

            if zone is None:
                # left the edge → reset
                current_zone = None
                zone_entry_time = 0.0
            else:
                # entered an edge zone
                if current_zone != zone:
                    current_zone = zone
                    zone_entry_time = now
                else:
                    # staying in same zone → check dwell time and cooldown
                    if (now - zone_entry_time >= DWELL_REQUIRED) and (now - last_action_time >= ACTION_COOLDOWN):
                        if current_zone == 'L':
                            print("Dwell LEFT 0.5s → Previous track")
                            try:
                                sp.previous_track()
                            except Exception:
                                pass
                        elif current_zone == 'R':
                            print("Dwell RIGHT 0.5s → Next track")
                            try:
                                sp.next_track()
                            except Exception:
                                pass
                        last_action_time = now
                        # require leaving the zone before another trigger
                        current_zone = None
                        zone_entry_time = 0.0
    else:
        # No hand → reset timers/state
        palm_open_time = 0.0
        current_zone = None
        zone_entry_time = 0.0

    # ----- Spotify-like panel (top-right) -----
    title, artist, album_cover, progress_ms, duration_ms = get_current_song_info()
    draw_top_left_panel(frame, title, artist, album_cover, progress_ms, duration_ms)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):  # ESC or q
        break

cap.release()
cv2.destroyAllWindows()
