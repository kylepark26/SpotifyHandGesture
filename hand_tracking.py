import os
import cv2
import mediapipe as mp
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import time
import requests
import numpy as np
from dotenv import load_dotenv

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0
)

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    for idx in (1, 2, 3):
        cap.release()
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            break
    if not cap.isOpened():
        raise RuntimeError("Could not open a camera. Disable Continuity Camera or try other indices.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 60)

cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hand Tracking", 1280, 800)

load_dotenv()

CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")

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

HOLD_TIME = 1.5
music_paused = False
palm_hold_start = 0.0
two_finger_left_start = 0.0
two_finger_right_start = 0.0
two_finger_up_start = 0.0
two_finger_down_start = 0.0
current_volume = 50
last_action_time = 0.0
ACTION_COOLDOWN = 2.0

frame_count = 0
spotify_info_cache = (None, None, None, None, None)
last_spotify_update = 0.0
SPOTIFY_UPDATE_INTERVAL = 1.0
cached_album_cover = None
last_track_id = None

fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0

last_gesture = ""
gesture_display_time = 0.0
GESTURE_DISPLAY_DURATION = 1.5

def is_palm_open(hand_landmarks):
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

def is_two_fingers_extended(hand_landmarks):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    def d(a, b):
        return np.hypot(a.x - b.x, a.y - b.y)

    index_extended = d(wrist, index_tip) > 0.20
    middle_extended = d(wrist, middle_tip) > 0.20
    ring_curled = d(ring_tip, ring_mcp) < 0.15
    pinky_curled = d(pinky_tip, pinky_mcp) < 0.15

    return index_extended and middle_extended and ring_curled and pinky_curled

def get_two_finger_direction(hand_landmarks, frame_width, frame_height):
    if not is_two_fingers_extended(hand_landmarks):
        return None

    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    avg_x = (index_tip.x + middle_tip.x) / 2
    avg_y = (index_tip.y + middle_tip.y) / 2

    LEFT_ZONE = 0.25
    RIGHT_ZONE = 0.75
    UP_ZONE = 0.30
    DOWN_ZONE = 0.70

    if avg_x < LEFT_ZONE:
        return 'left'
    elif avg_x > RIGHT_ZONE:
        return 'right'
    elif avg_y < UP_ZONE:
        return 'up'
    elif avg_y > DOWN_ZONE:
        return 'down'

    return None

def get_current_song_info():
    global cached_album_cover, last_track_id

    current = sp.current_playback()
    if current and current.get('item'):
        track = current['item']
        track_id = track['id']
        title = track['name']
        artist = ', '.join([a['name'] for a in track['artists']])
        duration_ms = track['duration_ms']
        progress_ms = current.get('progress_ms', 0)

        if track_id != last_track_id:
            try:
                album_cover_url = track['album']['images'][0]['url']
                response = requests.get(album_cover_url, timeout=1.0)
                arr = np.asarray(bytearray(response.content), dtype=np.uint8)
                cached_album_cover = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                last_track_id = track_id
            except Exception:
                pass

        return title, artist, cached_album_cover, progress_ms, duration_ms
    return None, None, None, None, None

def draw_top_left_panel(frame, title, artist, album_cover, progress_ms, duration_ms):
    h, w = frame.shape[:2]

    cover_size = 100
    panel_w = 320
    panel_h = cover_size + 40
    margin = 15
    x0, y0 = margin, margin
    x1, y1 = x0 + panel_w, y0 + panel_h

    SPOTIFY_BLACK = (18, 18, 18)
    SPOTIFY_GREEN = (84, 185, 29)
    TEXT_WHITE = (255, 255, 255)
    TEXT_GRAY = (179, 179, 179)
    PROGRESS_BG = (64, 64, 64)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), SPOTIFY_BLACK, -1)
    cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (50, 50, 50), 1)

    cover_x = x0 + 15
    cover_y = y0 + 15

    if album_cover is not None:
        ac = cv2.resize(album_cover, (cover_size, cover_size))
        frame[cover_y:cover_y+cover_size, cover_x:cover_x+cover_size] = ac
        cv2.rectangle(frame, (cover_x, cover_y),
                     (cover_x+cover_size, cover_y+cover_size), (80, 80, 80), 1)

    text_x = cover_x + cover_size + 15
    text_max_width = x1 - text_x - 15
    text_y_base = cover_y + 30

    if title:
        avg_char_width = 8
        max_chars = int(text_max_width / avg_char_width)
        title_short = title[:max_chars-3] + "..." if len(title) > max_chars else title
        cv2.putText(frame, title_short, (text_x, text_y_base),
                   cv2.FONT_HERSHEY_PLAIN, 1.3, TEXT_WHITE, 2, cv2.LINE_AA)

    if artist:
        avg_char_width = 7
        max_chars = int(text_max_width / avg_char_width)
        artist_short = artist[:max_chars-3] + "..." if len(artist) > max_chars else artist
        cv2.putText(frame, artist_short, (text_x, text_y_base + 25),
                   cv2.FONT_HERSHEY_PLAIN, 1.1, TEXT_GRAY, 1, cv2.LINE_AA)

    bar_y = cover_y + cover_size + 12
    bar_x0 = x0 + 15
    bar_x1 = x1 - 15
    bar_height = 3

    cv2.rectangle(frame, (bar_x0, bar_y - bar_height//2),
                 (bar_x1, bar_y + bar_height//2), PROGRESS_BG, -1, cv2.LINE_AA)

    if progress_ms is not None and duration_ms and duration_ms > 0:
        frac = max(0.0, min(1.0, progress_ms / float(duration_ms)))
        progress_x = int(bar_x0 + frac * (bar_x1 - bar_x0))

        cv2.rectangle(frame, (bar_x0, bar_y - bar_height//2),
                     (progress_x, bar_y + bar_height//2), SPOTIFY_GREEN, -1, cv2.LINE_AA)

        cv2.circle(frame, (progress_x, bar_y), 4, TEXT_WHITE, -1, cv2.LINE_AA)

        def mmss(ms):
            s = int(ms/1000)
            return f"{s//60}:{s%60:02d}"

        cv2.putText(frame, mmss(progress_ms), (bar_x0, bar_y + 15),
                   cv2.FONT_HERSHEY_PLAIN, 0.9, TEXT_GRAY, 1, cv2.LINE_AA)

        duration_text = mmss(duration_ms)
        duration_size = cv2.getTextSize(duration_text, cv2.FONT_HERSHEY_PLAIN, 0.9, 1)[0]
        cv2.putText(frame, duration_text, (bar_x1 - duration_size[0], bar_y + 15),
                   cv2.FONT_HERSHEY_PLAIN, 0.9, TEXT_GRAY, 1, cv2.LINE_AA)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w = frame.shape[:2]

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            now = time.time()

            if is_palm_open(lm):
                if palm_hold_start == 0.0:
                    palm_hold_start = now
                elif now - palm_hold_start >= HOLD_TIME and now - last_action_time >= ACTION_COOLDOWN:
                    current = sp.current_playback()
                    if current:
                        is_playing = current.get('is_playing')
                        if is_playing:
                            print("PAUSE")
                            last_gesture = "PAUSE"
                            gesture_display_time = now
                            sp.pause_playback()
                            music_paused = True
                        else:
                            print("PLAY")
                            last_gesture = "PLAY"
                            gesture_display_time = now
                            sp.start_playback()
                            music_paused = False
                    palm_hold_start = 0.0
                    last_action_time = now
            else:
                palm_hold_start = 0.0

            direction = get_two_finger_direction(lm, w, h)

            if direction == 'left':
                if two_finger_left_start == 0.0:
                    two_finger_left_start = now
                elif now - two_finger_left_start >= HOLD_TIME and now - last_action_time >= ACTION_COOLDOWN:
                    print("PREVIOUS TRACK")
                    last_gesture = "PREVIOUS"
                    gesture_display_time = now
                    try:
                        sp.previous_track()
                    except Exception as e:
                        print(f"Error: {e}")
                    two_finger_left_start = 0.0
                    last_action_time = now
            else:
                two_finger_left_start = 0.0

            if direction == 'right':
                if two_finger_right_start == 0.0:
                    two_finger_right_start = now
                elif now - two_finger_right_start >= HOLD_TIME and now - last_action_time >= ACTION_COOLDOWN:
                    print("NEXT TRACK")
                    last_gesture = "NEXT"
                    gesture_display_time = now
                    try:
                        sp.next_track()
                    except Exception as e:
                        print(f"Error: {e}")
                    two_finger_right_start = 0.0
                    last_action_time = now
            else:
                two_finger_right_start = 0.0

            if direction == 'up':
                if two_finger_up_start == 0.0:
                    two_finger_up_start = now
                elif now - two_finger_up_start >= HOLD_TIME and now - last_action_time >= ACTION_COOLDOWN:
                    current_volume = min(100, current_volume + 10)
                    print(f"Volume UP to {current_volume}%")
                    last_gesture = f"VOL {current_volume}%"
                    gesture_display_time = now
                    try:
                        sp.volume(current_volume)
                    except Exception as e:
                        print(f"Volume error: {e}")
                    two_finger_up_start = 0.0
                    last_action_time = now
            else:
                two_finger_up_start = 0.0

            if direction == 'down':
                if two_finger_down_start == 0.0:
                    two_finger_down_start = now
                elif now - two_finger_down_start >= HOLD_TIME and now - last_action_time >= ACTION_COOLDOWN:
                    current_volume = max(0, current_volume - 10)
                    print(f"Volume DOWN to {current_volume}%")
                    last_gesture = f"VOL {current_volume}%"
                    gesture_display_time = now
                    try:
                        sp.volume(current_volume)
                    except Exception as e:
                        print(f"Volume error: {e}")
                    two_finger_down_start = 0.0
                    last_action_time = now
            else:
                two_finger_down_start = 0.0

    else:
        palm_hold_start = 0.0
        two_finger_left_start = 0.0
        two_finger_right_start = 0.0
        two_finger_up_start = 0.0
        two_finger_down_start = 0.0

    fps_frame_count += 1
    if time.time() - fps_start_time >= 1.0:
        current_fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = time.time()

    now = time.time()
    if now - last_spotify_update >= SPOTIFY_UPDATE_INTERVAL:
        spotify_info_cache = get_current_song_info()
        last_spotify_update = now

    title, artist, album_cover, progress_ms, duration_ms = spotify_info_cache
    draw_top_left_panel(frame, title, artist, album_cover, progress_ms, duration_ms)

    fps_text = f"{current_fps} FPS"
    fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    fps_x = frame.shape[1] - fps_size[0] - 15
    fps_y = frame.shape[0] - 15

    overlay = frame.copy()
    cv2.rectangle(overlay, (fps_x - 8, fps_y - fps_size[1] - 8),
                 (fps_x + fps_size[0] + 8, fps_y + 8), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, fps_text, (fps_x, fps_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

    if last_gesture and (time.time() - gesture_display_time < GESTURE_DISPLAY_DURATION):
        h, w = frame.shape[:2]

        font_scale = 2.5
        text_size = cv2.getTextSize(last_gesture, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (h + text_size[1]) // 2

        overlay = frame.copy()
        padding = 40
        bg_x0 = text_x - padding
        bg_y0 = text_y - text_size[1] - padding
        bg_x1 = text_x + text_size[0] + padding
        bg_y1 = text_y + padding

        cv2.rectangle(overlay, (bg_x0, bg_y0), (bg_x1, bg_y1), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

        cv2.rectangle(frame, (bg_x0, bg_y0), (bg_x1, bg_y1), (84, 185, 29), 2)

        cv2.putText(frame, last_gesture, (text_x + 2, text_y + 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (84, 185, 29), 4, cv2.LINE_AA)
        cv2.putText(frame, last_gesture, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 4, cv2.LINE_AA)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
