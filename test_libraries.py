try:
    import mediapipe as mp
    print("MediaPipe imported successfully")
except ImportError as e:
    print(f"Error importing MediaPipe: {e}")

try:
    import cv2
    print("OpenCV imported successfully")
except ImportError as e:
    print(f"Error importing OpenCV: {e}")

try:
    import pyautogui
    print("PyAutoGUI imported successfully")
except ImportError as e:
    print(f"Error importing PyAutoGUI: {e}")

try:
    import spotipy
    print("Spotipy imported successfully")
except ImportError as e:
    print(f"Error importing Spotipy: {e}")