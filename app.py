# app.py (FINAL, COMPLETE, AND ROBUST CODE)

import os
import sys
import json
import base64
import numpy as np
import time
import random 

from flask import Flask, redirect, request, session, url_for, render_template

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.exceptions import SpotifyException

import cv2
from deepface import DeepFace
import mediapipe as mp

# --- CONFIGURATION ---
CLIENT_ID = "14585f7fdc0b410e911bd38719a874da"
CLIENT_SECRET = "ccfabae04403422b94203192f67790f2"
REDIRECT_URI = "http://127.0.0.1:8000/callback" 
SCOPE = "user-modify-playback-state user-read-playback-state user-read-currently-playing user-read-recently-played" 

# --- MOOD/GENRE MAPPING ---
MOOD_GENRE_MAP = {
    "happy": "happy", "joy": "happy", "😊": "happy", "😄": "happy", "🎉": "happy",
    "sad": "sad", "down": "sad", "😢": "sad", "😭": "sad", "💔": "sad",
    "angry": "rock", "mad": "rock", "😠": "rock", "😡": "rock", "🔥": "rock",
    "chill": "chill", "relaxed": "chill", "😌": "chill", "🧘": "chill",
    "party": "party", "dance": "party", "🥳": "party", "🕺": "party",
    "love": "romance", "romantic": "romance", "😍": "romance", "❤️": "romance",
    "workout": "workout", "gym": "workout", "💪": "workout", "🏋️": "workout",
    "sleep": "sleep", "tired": "sleep", "😴": "sleep", "🌙": "sleep",
}

EMOTION_GENRE_MAP = {
    "happy": "happy", "sad": "sad", "angry": "rock", "neutral": "chill",
    "surprise": "pop", "fear": "ambient", "disgust": "metal"
}

# --- GLOBAL STATE ---
MAX_PLAYLIST_SIZE = 15
GLOBAL_PLAYLIST = []
CURRENT_SONG_INDEX = -1 
LAST_DETECTED_EMOTION = None
LAST_DETECTED_TRACK = None
FACE_STATE = "waiting" 

# --- FLASK APP SETUP ---
app = Flask(__name__)
app.secret_key = os.urandom(64) 

def create_spotify_oauth():
    return SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=None 
    )

def get_spotify_client():
    token_info = session.get('token_info', None)
    if not token_info:
        return None
    
    sp_oauth = create_spotify_oauth()
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
        
    return spotipy.Spotify(auth=token_info['access_token'])

# --- REUSABLE SPOTIFY ACTION (Market="IN" Added) ---
def search_and_play(sp, genre):
    """Searches for a track by genre and starts playback, also populates GLOBAL_PLAYLIST."""
    global GLOBAL_PLAYLIST, CURRENT_SONG_INDEX
    
    try:
        # SEARCH MARKET MODIFIED: Set to "IN" for better regional results
        results = sp.search(q=f'genre:{genre}', type='track', limit=MAX_PLAYLIST_SIZE, market="IN")
        tracks = results['tracks']['items']

        if not tracks:
            return None, "Could not find any songs for the genre: " + genre, False

        GLOBAL_PLAYLIST = tracks
        CURRENT_SONG_INDEX = 0
        
        track_to_play = tracks[0]
        track_uri = track_to_play['uri']
        
        devices = sp.devices()
        if not devices['devices']:
             return None, "No active Spotify device found. Please open Spotify app on a device.", False
        
        sp.start_playback(uris=[track_uri])
        message = f"Playing: '{track_to_play['name']}' by {track_to_play['artists'][0]['name']}"
        
        return track_to_play, message, True

    except SpotifyException:
        return None, "Playback failed: Please ensure your Spotify app is running and active.", False
    except Exception as e:
        return None, f"An unexpected error occurred: {e}", False

# --- NEW: AUDIO FEATURE ANALYSIS FUNCTION ---
def analyze_audio_features(sp, track_id):
    """Guess the mood of a track based on its audio features (valence, energy) with relaxed thresholds."""
    try:
        features = sp.audio_features([track_id])[0]
        if not features:
            return "unknown", "Could not fetch audio features."

        valence = features.get('valence', 0.5) 
        energy = features.get('energy', 0.5)   
        
        # --- IMPROVED MOOD LOGIC (Relaxed Thresholds) ---
        
        # 1. High Energy / Positive (Happy / Party)
        if valence >= 0.65 and energy >= 0.6:
            return "happy", "High Energy, Positive Vibe."
        
        # 2. Low Energy / Positive/Neutral (Chill / Relaxed)
        elif valence >= 0.5 and energy <= 0.5:
            return "chill", "Relaxed, Calm, Positive/Neutral."
            
        # 3. High Energy / Negative/Neutral (Angry / Intense)
        elif valence <= 0.5 and energy >= 0.7:
            return "angry", "High Energy, Intense Vibe."
            
        # 4. Low Energy / Negative (Sad)
        elif valence <= 0.4 and energy <= 0.5:
            return "sad", "Low Energy, Somber Vibe."
            
        # 5. Default Fallback (Balanced or Moderate)
        else:
            return "neutral", "Balanced or Moderate Mood."

    except Exception as e:
        print(f"Error analyzing audio features: {e}")
        return "unknown", f"Analysis error: {e}"

# --- PLAY CONFIRMATION ROUTE (Dynamic Search + Error Handling) ---
@app.route('/play/confirmed', methods=['POST'])
def play_confirmed():
    global FACE_STATE, LAST_DETECTED_TRACK, LAST_DETECTED_EMOTION, GLOBAL_PLAYLIST, CURRENT_SONG_INDEX
    sp = get_spotify_client()
    if not sp:
        return {"status": "error", "message": "User not authenticated."}, 401
    
    action = request.json.get('action')
    
    if action == 'play' and FACE_STATE == 'detected' and LAST_DETECTED_TRACK:
        emotion = LAST_DETECTED_EMOTION
        
        # 1. Determine matching tracks from GLOBAL_PLAYLIST (currently playing context)
        emotion_genre = EMOTION_GENRE_MAP.get(emotion, "pop")
        
        # Filter based on broad keywords in name/artist matching the emotion/genre
        matching_tracks = [
            track for track in GLOBAL_PLAYLIST 
            if track and (any(keyword in track.get('name', '').lower() for keyword in [emotion_genre, emotion])) or
               (any(keyword in artist['name'].lower() for artist in track.get('artists', [])) for keyword in [emotion_genre, emotion])
        ]
        
        track_to_play = random.choice(matching_tracks) if matching_tracks else LAST_DETECTED_TRACK

        try:
            sp.start_playback(uris=[track_to_play['uri']])
            
            # Find the index of the played track (optional, for Next/Previous coherence)
            try:
                CURRENT_SONG_INDEX = GLOBAL_PLAYLIST.index(track_to_play)
            except ValueError:
                CURRENT_SONG_INDEX = 0

            FACE_STATE = "playing"
            message = f"CONFIRMED! Playing: '{track_to_play['name']}' by {track_to_play['artists'][0]['name']} (Context Match: {emotion})"
            return {"status": "success", "message": message, "new_state": FACE_STATE}
        
        except SpotifyException:
            return {"status": "error", "message": "Playback failed: Ensure Spotify app is running and active.", "new_state": FACE_STATE}
        except Exception as e:
            return {"status": "error", "message": f"Playback error: {e}", "new_state": FACE_STATE}

    elif action == 'reset':
        FACE_STATE = "waiting"
        LAST_DETECTED_EMOTION = None
        LAST_DETECTED_TRACK = None
        return {"status": "success", "message": "Detection reset. Waiting for new expression.", "new_state": FACE_STATE}

    return {"status": "error", "message": "Invalid state or action."}

# --- NEW ROUTE: ANALYZE CURRENT SONG (Market="IN" Added) ---
@app.route('/analyze/current_track', methods=['POST'])
def analyze_current_track():
    sp = get_spotify_client()
    if not sp:
        return {"status": "error", "message": "User not authenticated."}, 401
    
    try:
        current_playback = sp.current_playback()
        if not current_playback or not current_playback.get('is_playing'):
            return {"status": "info", "message": "No song currently playing on Spotify."}

        current_track = current_playback['item']
        track_id = current_track['id']
        track_name = current_track['name']
        artist_name = current_track['artists'][0]['name']

        mood, analysis_note = analyze_audio_features(sp, track_id)

        suggested_emotion = mood 
        suggested_genre = EMOTION_GENRE_MAP.get(suggested_emotion, 'pop')
        
        # SEARCH MARKET MODIFIED: Set to "IN" for better regional results
        search_query = f"{suggested_emotion} genre:{suggested_genre}"
        results = sp.search(q=search_query, type='track', limit=1, market="IN")
        suggested_track = results['tracks']['items'][0] if results['tracks']['items'] else None
        
        if suggested_track:
            suggestion_message = f"For this '{mood.upper()}' song, we suggest: '{suggested_track['name']}' by {suggested_track['artists'][0]['name']}."
        else:
            suggestion_message = f"The song's mood is '{mood.upper()}'. No direct suggestion found."

        return {
            "status": "success",
            "message": f"Analyzed '{track_name}' by {artist_name}. Mood: {mood.upper()}.",
            "analyzed_mood": mood,
            "suggestion": suggestion_message,
            "analysis_note": analysis_note
        }

    except Exception as e:
        return {"status": "error", "message": f"Error analyzing current song: {e}"}

# app.py (Modified Routes Section)

# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Checks authentication status and redirects to controller or login page."""
    # If session has token info, user is authenticated: show the controller
    if 'token_info' in session:
        return render_template('index_v2.html')
    
    # Otherwise, show the dedicated login page
    return render_template('login.html')

@app.route('/spotify_login')
def spotify_login():
    """Initiates the Spotify OAuth process."""
    sp_oauth = create_spotify_oauth()
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route('/callback')
def callback():
    """Spotify redirects here after successful login."""
    sp_oauth = create_spotify_oauth()
    session.clear()
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    # Redirect back to the main index route, which will now show the controller
    return redirect(url_for('index'))

@app.route('/logout')
def logout():
    """Clears the session and redirects to the login page."""
    session.clear()
    return redirect(url_for('index'))

@app.route('/recommend/text', methods=['POST'])
def recommend_text():
    global GLOBAL_PLAYLIST # Ensure we access the global list
    sp = get_spotify_client()
    if not sp:
        return {"status": "error", "message": "User not authenticated."}, 401
    
    user_input = request.form.get('mood', '').lower().strip()
    genre = MOOD_GENRE_MAP.get(user_input)
    
    if not genre:
        return {"status": "error", "message": "Invalid mood/emoji. Please choose a valid option."}, 400

    track, message, success = search_and_play(sp, genre)
    
    # MODIFIED: Return the GLOBAL_PLAYLIST data along with the status
    return {
        "status": "success" if success else "error", 
        "message": message, 
        "genre": genre,
        "playlist": GLOBAL_PLAYLIST # Send the playlist to the frontend
    }

# --- VISION AND GESTURE SETUP ---
mp_hands = mp.solutions.hands
hands_model = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
last_action_time = 0
COOLDOWN_SECONDS = 1.5 
tip_ids = [4, 8, 12, 16, 20]
LAST_WRIST_Y = 0.5 # Added for vertical gesture tracking

# --- GESTURE PROCESSING HELPER FUNCTION (Modified Error Handling) ---
def process_gesture_action(frame, sp):
    """Checks for hand gestures (Next/Previous/Play/Stop) and executes them if valid."""
    global last_action_time, GLOBAL_PLAYLIST, CURRENT_SONG_INDEX, FACE_STATE, LAST_WRIST_Y
    
    # Define vertical movement threshold
    Y_SWIPE_THRESHOLD = 0.15 

    current_time = time.time()
    if (current_time - last_action_time) < COOLDOWN_SECONDS:
         return {"status": "cooldown", "message": f"Gesture cooldown. Time left: {COOLDOWN_SECONDS - (current_time - last_action_time):.1f}s"}

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb_frame)
    
    action = None
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x 
        current_wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        
        # Calculate vertical difference (Downward movement is positive delta)
        Y_DELTA = current_wrist_y - LAST_WRIST_Y

        # 1. Check for Next/Previous (Horizontal Position)
        if wrist_x > 0.7:
            action = "next"
        elif wrist_x < 0.3:
            action = "previous"
        
        # 2. Check for STOP gesture (Fast Downward Swipe in center/neutral zone)
        elif Y_DELTA > Y_SWIPE_THRESHOLD and 0.3 < wrist_x < 0.7:
            action = "stop"
            
        else:
            # 3. Check for Open Hand (Play/Load Context)
            fingers = []
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x: fingers.append(1)
            else: fingers.append(0)
            for id in tip_ids[1:]:
                if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y: fingers.append(1)
                else: fingers.append(0)
            if sum(fingers) == 5: action = "play" 
        
        # Update last wrist Y position for the next frame's comparison
        LAST_WRIST_Y = current_wrist_y


        if action:
            
            # --- ACTION EXECUTION ---
            
            if action == "stop":
                try:
                    sp.pause_playback()
                    last_action_time = current_time
                    FACE_STATE = "waiting" # Reset face detection state
                    return {"status": "gesture_success", "message": "Playback Paused (STOP Gesture).", "action": action}
                except SpotifyException:
                    return {"status": "gesture_error", "message": "Playback failed. Ensure Spotify app is open and playing music.", "action": action}
                except Exception as e:
                    return {"status": "gesture_error", "message": f"An unexpected error occurred during STOP: {e}", "action": action}
            
            if action == "play":
                # Special logic for 'Play': Fetch current Spotify context if playlist is empty
                if not GLOBAL_PLAYLIST:
                    try:
                        playback = sp.current_playback()
                        if playback and playback.get('context') and playback['context'].get('uri'):
                            context_uri = playback['context']['uri']
                            # Fetch items from the playlist/album context URI
                            if 'playlist' in context_uri or 'album' in context_uri:
                                items = sp.playlist_items(context_uri, limit=MAX_PLAYLIST_SIZE).get('items', []) if 'playlist' in context_uri else sp.album_tracks(context_uri, limit=MAX_PLAYLIST_SIZE).get('items', [])
                                
                                GLOBAL_PLAYLIST = [item.get('track', item) for item in items if item.get('track') or item]

                            if not GLOBAL_PLAYLIST:
                                track, message, success = search_and_play(sp, "chill")
                                return {"status": "gesture_success", "message": "Context empty. Starting Chill playlist.", "action": "New Playlist (Play)"}
                        
                        else:
                             track, message, success = search_and_play(sp, "chill")
                             return {"status": "gesture_success", "message": "No active context. Starting Chill playlist.", "action": "New Playlist (Play)"}
                    except Exception:
                        track, message, success = search_and_play(sp, "chill")
                        return {"status": "gesture_success", "message": "Error fetching context. Starting Chill playlist.", "action": "New Playlist (Play)"}
                
                CURRENT_SONG_INDEX = 0 
            
            elif action == "next" and GLOBAL_PLAYLIST:
                CURRENT_SONG_INDEX = (CURRENT_SONG_INDEX + 1) % len(GLOBAL_PLAYLIST)

            elif action == "previous" and GLOBAL_PLAYLIST:
                CURRENT_SONG_INDEX = (CURRENT_SONG_INDEX - 1) % len(GLOBAL_PLAYLIST)
            
            else:
                return {"status": "info", "message": "Gesture detected but no playlist exists. Use Open Hand or Text Search first."}
            
            track_to_play = GLOBAL_PLAYLIST[CURRENT_SONG_INDEX]
            track_uri = track_to_play['uri']
            
            try:
                sp.start_playback(uris=[track_uri])
                last_action_time = current_time
                FACE_STATE = "waiting" 
                message = f"Song: '{track_to_play['name']}' by {track_to_play['artists'][0]['name']}"
                return {"status": "gesture_success", "message": message, "action": action}
            except SpotifyException:
                return {"status": "gesture_error", "message": "Playback failed. Ensure Spotify app is open and playing music.", "action": action}
            except Exception as e:
                return {"status": "gesture_error", "message": f"An unexpected playback error occurred: {e}", "action": action}

    return None 

# --- VISION RECOMMENDER ROUTE (Face/Gesture Combined) ---
@app.route('/recommend/vision', methods=['POST'])
def recommend_vision():
    global FACE_STATE, LAST_DETECTED_EMOTION, LAST_DETECTED_TRACK, GLOBAL_PLAYLIST
    sp = get_spotify_client()
    if not sp:
        return {"status": "error", "message": "User not authenticated."}, 401
    
    data = request.json
    base64_img = data.get('frame')
    mode = data.get('mode')
    
    # ... (Frame decoding logic) ...
    try:
        img_bytes = base64.b64decode(base64_img.split(',')[1])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame = cv2.flip(frame, 1) 
    except Exception as e:
        # NOTE: This error is handled by JS frontend validation now, but kept for backend robustness.
        return {"status": "error", "message": f"Frame decoding error: {e}"}, 400

    # --- SIMULTANEOUS FACE + GESTURE LOGIC ---
    
    # 1. Process Gestures First (Override)
    gesture_result = process_gesture_action(frame, sp)
    if gesture_result and gesture_result['status'].startswith('gesture'):
        FACE_STATE = "waiting"
        LAST_DETECTED_EMOTION = None
        LAST_DETECTED_TRACK = None
        return {"status": gesture_result['status'], "message": gesture_result['message'], "action": gesture_result['action'], "new_state": FACE_STATE}
    elif gesture_result and gesture_result['status'] == 'cooldown':
        pass 
    
    # 2. Process Facial Expression 
    if mode == 'face':
        if FACE_STATE == "detected":
            track = LAST_DETECTED_TRACK
            message = f"Detected: {LAST_DETECTED_EMOTION.upper()}. Confirm to play: '{track['name']}'"
            return {"status": "pending", "message": message, "emotion": LAST_DETECTED_EMOTION, "new_state": FACE_STATE}
        
        elif FACE_STATE == "playing":
            pass 
        
        # If FACE_STATE is "waiting" or "playing", run new detection
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            if isinstance(result, list) and len(result) > 0:
                dominant_emotion = result[0]['dominant_emotion']
                
                if dominant_emotion != LAST_DETECTED_EMOTION or not LAST_DETECTED_TRACK: 
                    
                    emotion_genre = EMOTION_GENRE_MAP.get(dominant_emotion, "pop")
                    
                    # Search local GLOBAL_PLAYLIST by track name/artist keywords 
                    matching_tracks = [
                        track for track in GLOBAL_PLAYLIST 
                        if track and (any(keyword in track.get('name', '').lower() for keyword in [emotion_genre, dominant_emotion])) or
                           (any(keyword in artist['name'].lower() for artist in track.get('artists', [])) for keyword in [emotion_genre, dominant_emotion])
                    ]
                    
                    if matching_tracks:
                        suggested_track = random.choice(matching_tracks)
                        
                        LAST_DETECTED_TRACK = suggested_track
                        LAST_DETECTED_EMOTION = dominant_emotion
                        FACE_STATE = "detected" 
                        
                        track_name = LAST_DETECTED_TRACK['name']
                        message = f"Detected {dominant_emotion.upper()}! Suggested from playlist: '{track_name}'. Click PLAY to confirm."
                        return {"status": "detected", "message": message, "emotion": dominant_emotion, "new_state": FACE_STATE}
                    
                    else:
                        return {"status": "info", "message": f"Detected {dominant_emotion.upper()}. Use Open Hand gesture first to load context playlist!"}
        
        except Exception:
            return {"status": "info", "message": "No face detected."}

    elif mode == 'gesture':
        return {"status": "info", "message": "Gesture mode running. Use hand motions."}
        
    return {"status": "info", "message": "No face or hand detected."}


if __name__ == '__main__':
    try:
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"Failed to start Flask application: {e}")