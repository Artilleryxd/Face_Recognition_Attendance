import os
import cv2
import numpy as np
import face_recognition
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
import time

app = Flask(__name__, template_folder='app/templates', static_folder='app/static')
app.secret_key = 'biometric_attendance_system'

# Ensure directories exist
os.makedirs('data/faces', exist_ok=True)
os.makedirs('attendance csv', exist_ok=True)

# Global variables
known_face_encodings = []
known_face_names = []
face_locations = []
face_encodings = []
face_names = []
students_info = {}

# Load student data
def load_student_data():
    global known_face_encodings, known_face_names, students_info
    known_face_encodings = []
    known_face_names = []
    students_info = {}
    
    if os.path.exists('data/students.csv'):
        df = pd.read_csv('data/students.csv')
        for _, row in df.iterrows():
            student_id = row['student_id']
            name = row['name']
            
            # Load face encoding if it exists
            encoding_path = f'data/faces/{student_id}.npy'
            if os.path.exists(encoding_path):
                encoding = np.load(encoding_path)
                known_face_encodings.append(encoding)
                known_face_names.append(student_id)
                students_info[student_id] = name

# Initialize the system
def initialize_system():
    global students_info
    
    # Create students.csv if it doesn't exist
    if not os.path.exists('data/students.csv'):
        df = pd.DataFrame(columns=['student_id', 'name'])
        df.to_csv('data/students.csv', index=False)
    
    load_student_data()

# Mark attendance
def mark_attendance(student_id):
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d')
    time_string = now.strftime('%H:%M:%S')
    
    # Create or append to attendance CSV for the day
    attendance_file = f'attendance csv/attendance_{date_string}.csv'
    
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=['student_id', 'name', 'time'])
        df.to_csv(attendance_file, index=False)
    
    # Check if student already marked attendance today
    df = pd.read_csv(attendance_file)
    if not df['student_id'].eq(student_id).any():
        student_name = students_info.get(student_id, "Unknown")
        new_row = {'student_id': student_id, 'name': student_name, 'time': time_string}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(attendance_file, index=False)
        return True
    return False

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        
        # Simple validation
        if not student_id or not name:
            flash('Student ID and Name are required!', 'danger')
            return redirect(url_for('register'))
        
        # Check if student ID already exists
        if os.path.exists('data/students.csv'):
            df = pd.read_csv('data/students.csv')
            if student_id in df['student_id'].values:
                flash('Student ID already exists!', 'danger')
                return redirect(url_for('register'))
        
        # Add student to CSV
        new_student = {'student_id': student_id, 'name': name}
        df = pd.read_csv('data/students.csv')
        df = pd.concat([df, pd.DataFrame([new_student])], ignore_index=True)
        df.to_csv('data/students.csv', index=False)
        
        # Reload student data
        load_student_data()
        
        flash('Student registered successfully! Now capture the face.', 'success')
        return redirect(url_for('capture_face', student_id=student_id))
    
    return render_template('register.html')

@app.route('/capture_face/<student_id>')
def capture_face(student_id):
    return render_template('capture_face.html', student_id=student_id)

# Video feed for face capture
@app.route('/video_feed_capture/<student_id>')
def video_feed_capture(student_id):
    def generate_frames_capture(student_id):
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                # Create an error image to display when camera is not accessible
                error_img = np.zeros((300, 500, 3), dtype=np.uint8)
                cv2.putText(error_img, "ERROR: Camera access denied", (30, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_img)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                return
                
            face_captured = False
            countdown = 50  # frames to wait before capturing
            
            while not face_captured:
                success, frame = camera.read()
                if not success:
                    break
                    
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations_capture = face_recognition.face_locations(rgb_frame)
                
                if face_locations_capture:
                    if countdown > 0:
                        # Display countdown
                        cv2.putText(frame, f"Hold still... {countdown//10}", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        countdown -= 1
                    else:
                        # Capture face
                        face_encodings_capture = face_recognition.face_encodings(rgb_frame, face_locations_capture)
                        if face_encodings_capture:
                            # Save encoding
                            np.save(f'data/faces/{student_id}.npy', face_encodings_capture[0])
                            
                            # Reload student data
                            load_student_data()
                            
                            # Indicate capture completed
                            cv2.putText(frame, "Face captured successfully!", (50, 50), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            face_captured = True
                            
                            # Show the frame with success message for a moment
                            ret, buffer = cv2.imencode('.jpg', frame)
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            
                            # Wait a bit to show the success message
                            time.sleep(2)
                            break
                
                # Draw box around face
                for (top, right, bottom, left) in face_locations_capture:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            camera.release()
        
        except Exception as e:
            # Create an error image to display when exceptions occur
            error_img = np.zeros((300, 500, 3), dtype=np.uint8)
            cv2.putText(error_img, f"ERROR: {str(e)}", (30, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames_capture(student_id), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_captured/<student_id>')
def face_captured(student_id):
    return jsonify({'success': os.path.exists(f'data/faces/{student_id}.npy')})

@app.route('/attendance')
def attendance():
    return render_template('attendance.html')

# Video feed for attendance
@app.route('/video_feed_attendance')
def video_feed_attendance():
    def generate_frames_attendance():
        try:
            camera = cv2.VideoCapture(1)
            if not camera.isOpened():
                # Create an error image to display when camera is not accessible
                error_img = np.zeros((300, 500, 3), dtype=np.uint8)
                cv2.putText(error_img, "ERROR: Camera access denied", (30, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                ret, buffer = cv2.imencode('.jpg', error_img)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                return
                
            process_this_frame = True
            last_recognition_time = datetime.now()
            recognition_cooldown = 5  # seconds
            
            while True:
                success, frame = camera.read()
                if not success:
                    break
                    
                # Only process every other frame to save time
                if process_this_frame:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Find faces
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for known faces
                        matches = []
                        if known_face_encodings:
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        
                        name = "Unknown"
                        
                        if True in matches:
                            # Get the first matching face
                            matched_index = matches.index(True)
                            name = known_face_names[matched_index]
                            
                            # Check if enough time has passed since last recognition
                            current_time = datetime.now()
                            if (current_time - last_recognition_time).total_seconds() > recognition_cooldown:
                                # Mark attendance
                                attendance_marked = mark_attendance(name)
                                if attendance_marked:
                                    # Show message on frame
                                    cv2.putText(frame, f"Attendance marked: {students_info.get(name, name)}", 
                                              (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                                last_recognition_time = current_time
                        
                        face_names.append(name)
                
                process_this_frame = not process_this_frame
                
                # Display results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Draw box around face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Draw label with name
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, students_info.get(name, name), (left + 6, bottom - 6), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        except Exception as e:
            # Create an error image to display when exceptions occur
            error_img = np.zeros((300, 500, 3), dtype=np.uint8)
            cv2.putText(error_img, f"ERROR: {str(e)}", (30, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', error_img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return Response(generate_frames_attendance(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/view_attendance')
def view_attendance():
    attendance_files = []
    attendance_dir = 'attendance csv'
    
    for file in os.listdir(attendance_dir):
        if file.startswith('attendance_') and file.endswith('.csv'):
            date_str = file.replace('attendance_', '').replace('.csv', '')
            attendance_files.append({'date': date_str, 'filename': file})
    
    # Sort by date (most recent first)
    attendance_files.sort(key=lambda x: x['date'], reverse=True)
    
    return render_template('view_attendance.html', attendance_files=attendance_files)

@app.route('/attendance_data/<filename>')
def attendance_data(filename):
    filepath = os.path.join('attendance csv', filename)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return jsonify(df.to_dict(orient='records'))
    return jsonify([])

@app.route('/students')
def students():
    if os.path.exists('data/students.csv'):
        df = pd.read_csv('data/students.csv')
        students_list = df.to_dict(orient='records')
        
        # Add face captured status
        for student in students_list:
            student_id = student['student_id']
            student['face_captured'] = os.path.exists(f'data/faces/{student_id}.npy')
        
        return render_template('students.html', students=students_list)
    return render_template('students.html', students=[])

# Initialize system on startup
initialize_system()

if __name__ == '__main__':
    app.run(debug=True)