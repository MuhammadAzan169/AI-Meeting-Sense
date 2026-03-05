import cv2
import time
import numpy as np
import os
import insightface
import pandas as pd
from datetime import datetime
import threading
import keyboard

# === Configuration === #
VIDEO_SOURCE = 0
SIMILARITY_THRESHOLD = 0.4
CSV_FILENAME = "attendance_sessions.csv"

# Global variables
known_embeddings = []
known_names = []
all_required_persons = set()
sessions = []  # List of session records
current_session = None
is_capturing = False
session_counter = 0

# Add a lock for thread safety
session_lock = threading.Lock()

# === Known faces loader === #
def load_known_faces(known_faces_dir='Faces'):
    model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
    model.prepare(ctx_id=0)
    known_embeddings, known_names = [], []
    
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        all_required_persons.add(person_name)
            
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
                
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            faces = model.get(img)
            if faces:
                known_embeddings.append(faces[0].embedding)
                known_names.append(person_name)
                
    print(f"[INFO] Loaded {len(known_embeddings)} known faces from {len(all_required_persons)} persons.")
    return known_embeddings, known_names

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_faces_in_frame(frame, face_model):
    """Recognize faces in a single frame and return recognized names"""
    faces = face_model.get(frame)
    recognized_names = set()
    
    if faces:
        for face in faces:
            embedding = face.embedding
            name = "Unknown"
            max_sim = SIMILARITY_THRESHOLD
            
            for known_emb, known_name in zip(known_embeddings, known_names):
                sim = cosine_similarity(embedding, known_emb)
                if sim > max_sim:
                    max_sim = sim
                    name = known_name
            
            if name != "Unknown":
                recognized_names.add(name)
                
    return recognized_names, faces

def start_session():
    """Start a new attendance session"""
    global current_session, session_counter, is_capturing
    
    with session_lock:
        if is_capturing:
            print("[INFO] Session is already running. Stop current session first.")
            return
        
        session_counter += 1
        current_session = {
            'session_id': session_counter,
            'start_time': datetime.now(),
            'end_time': None,
            'present_persons': set(),
            'frame_count': 0
        }
        is_capturing = True
        print(f"\n[SESSION {session_counter}] Started at {current_session['start_time'].strftime('%H:%M:%S')}")
        print(f"[INFO] Session {session_counter} capturing... Press 's' to stop.")

def stop_session():
    """Stop the current attendance session and record results"""
    global current_session, is_capturing
    
    with session_lock:
        if not is_capturing or current_session is None:
            print("[INFO] No active session to stop.")
            return
        
        current_session['end_time'] = datetime.now()
        current_session['duration'] = (current_session['end_time'] - current_session['start_time']).total_seconds()
        
        # Create session record
        session_record = {
            'Session_ID': current_session['session_id'],
            'Date': current_session['start_time'].date().isoformat(),
            'Start_Time': current_session['start_time'].strftime('%H:%M:%S'),
            'End_Time': current_session['end_time'].strftime('%H:%M:%S'),
            'Duration_Seconds': current_session['duration'],
            'Duration_Formatted': f"{int(current_session['duration'] // 3600)}h {int((current_session['duration'] % 3600) // 60)}m {int(current_session['duration'] % 60)}s",
            'Total_Frames': current_session['frame_count']
        }
        
        # Add attendance for each person
        for person in sorted(all_required_persons):
            if person in current_session['present_persons']:
                session_record[person] = 'Present'
            else:
                session_record[person] = 'Absent'
        
        sessions.append(session_record)
        
        # Print session summary
        print(f"\n=== SESSION {session_counter} SUMMARY ===")
        print(f"Duration: {session_record['Duration_Formatted']}")
        print(f"Total Frames: {current_session['frame_count']}")
        
        present_count = len(current_session['present_persons'])
        absent_count = len(all_required_persons) - present_count
        print(f"Present: {present_count}/{len(all_required_persons)}")
        print(f"Absent: {absent_count}/{len(all_required_persons)}")
        
        if present_count > 0:
            print("\nPresent persons:")
            for person in sorted(current_session['present_persons']):
                print(f"  ✓ {person}")
        
        if absent_count > 0:
            print("\nAbsent persons:")
            for person in sorted(all_required_persons - current_session['present_persons']):
                print(f"  ✗ {person}")
        
        # Reset for next session
        is_capturing = False
        current_session = None
        print(f"\n[INFO] Session {session_counter} ended. Ready for next session.")

def save_sessions_to_csv():
    """Save all recorded sessions to CSV file with proper structure (Names as rows)"""
    if not sessions:
        print("[INFO] No sessions recorded.")
        return
    
    # Create DataFrame in the desired format
    # Each person is a row, each session is a column
    rows = []
    
    for person in sorted(all_required_persons):
        row = {'Name': person}
        for session in sessions:
            session_id = session['Session_ID']
            row[f'Session_{session_id}'] = session.get(person, 'Absent')
        rows.append(row)
    
    # Create summary row for each session
    summary_row = {'Name': 'SUMMARY'}
    for session in sessions:
        session_id = session['Session_ID']
        present_count = sum(1 for person in all_required_persons if session.get(person) == 'Present')
        summary_row[f'Session_{session_id}'] = f"{present_count}/{len(all_required_persons)} Present"
    rows.append(summary_row)
    
    # Create metadata rows
    for session in sessions:
        session_id = session['Session_ID']
        # Date row
        date_row = {'Name': f'Session_{session_id}_Date'}
        date_row[f'Session_{session_id}'] = session['Date']
        rows.append(date_row)
        
        # Time row
        time_row = {'Name': f'Session_{session_id}_Time'}
        time_row[f'Session_{session_id}'] = f"{session['Start_Time']} to {session['End_Time']}"
        rows.append(time_row)
        
        # Duration row
        duration_row = {'Name': f'Session_{session_id}_Duration'}
        duration_row[f'Session_{session_id}'] = session['Duration_Formatted']
        rows.append(duration_row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    df.to_csv(CSV_FILENAME, index=False)
    
    print(f"\n=== ALL SESSIONS SAVED ===")
    print(f"Total Sessions: {len(sessions)}")
    print(f"File: {CSV_FILENAME}")
    
    # Print summary table
    print("\nSession Summary:")
    for session in sessions:
        session_id = session['Session_ID']
        present_count = sum(1 for person in all_required_persons if session.get(person) == 'Present')
        print(f"Session {session_id}: {session['Start_Time']} to {session['End_Time']} - {present_count}/{len(all_required_persons)} present")

def keyboard_listener():
    """Listen for keyboard commands in a separate thread"""
    print("\n=== KEYBOARD CONTROLS ===")
    print("'s' - Start/Stop session (toggle)")
    print("'q' - Quit and save all sessions")
    print("==========================\n")
    
    while True:
        try:
            if keyboard.is_pressed('s'):
                if not is_capturing:
                    start_session()
                else:
                    stop_session()
                time.sleep(0.5)  # Debounce
            
            if keyboard.is_pressed('q'):
                print("\n[INFO] Quit requested...")
                if is_capturing:
                    print("[INFO] Stopping active session before quitting...")
                    stop_session()
                break
                
            time.sleep(0.1)
        except Exception as e:
            print(f"[ERROR] Keyboard listener: {e}")
            break

def main():
    global known_embeddings, known_names
    
    # Load known faces
    print("[INFO] Loading known faces...")
    known_embeddings, known_names = load_known_faces()
    
    if not all_required_persons:
        print("[ERROR] No known persons found in KnownFaces directory.")
        return
    
    print(f"\n[INFO] System ready for {len(all_required_persons)} persons:")
    for person in sorted(all_required_persons):
        print(f"  - {person}")
    
    # Initialize face model
    print("\n[INFO] Loading face recognition model...")
    face_model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
    face_model.prepare(ctx_id=0)
    
    # Start keyboard listener in separate thread
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()
    
    # Open camera
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return
    
    print("\n[INFO] Camera started. Waiting for session commands...")
    print("[INFO] Press 's' to start first session.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed.")
                continue
            
            display_frame = frame.copy()
            faces_to_draw = []
            
            # If session is active, process frame for attendance
            with session_lock:
                if is_capturing and current_session:
                    current_session['frame_count'] += 1
                    
                    # Recognize faces in current frame using InsightFace
                    recognized_names, faces = recognize_faces_in_frame(frame, face_model)
                    
                    # Update present persons
                    current_session['present_persons'].update(recognized_names)
                    
                    # Store faces for drawing
                    faces_to_draw = faces if faces else []
            
            # Draw recognized faces
            for face in faces_to_draw:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display session status
            with session_lock:
                if is_capturing and current_session:
                    status_text = f"SESSION {current_session['session_id']} - ACTIVE"
                    status_color = (0, 255, 0)
                    
                    # Show recognized count
                    present_count = len(current_session['present_persons'])
                    cv2.putText(display_frame, f"Present: {present_count}/{len(all_required_persons)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Show recognized persons
                    y_offset = 90
                    for person in sorted(current_session['present_persons']):
                        cv2.putText(display_frame, f"✓ {person}", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 25
                else:
                    status_text = "SESSION INACTIVE - Press 's' to start"
                    status_color = (0, 0, 255)
            
            # Show session status
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Show instructions
            cv2.putText(display_frame, "Press 's': Start/Stop Session", 
                       (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press 'q': Quit & Save", 
                       (10, display_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Attendance System - Session Based", display_frame)
            
            # Check for quit from main thread
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested from main thread...")
                if is_capturing:
                    print("[INFO] Stopping active session before quitting...")
                    stop_session()
                break
            
            # Small delay to prevent high CPU usage
            time.sleep(0.03)
            
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Save all sessions to CSV
        if sessions:
            save_sessions_to_csv()
        else:
            print("[INFO] No sessions were recorded.")
        
        print("\n[INFO] System shutdown complete.")

if __name__ == "__main__":
    main()