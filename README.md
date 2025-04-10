# Face Recognition Biometric Attendance System

A Python-based application that uses facial recognition to automatically mark attendance and maintain records in CSV format.

## Features

- **Student Registration**: Add new students with their details
- **Face Capture**: Capture student faces for recognition
- **Attendance Marking**: Automatically recognize students and mark attendance
- **Attendance Records**: View and track attendance by date
- **CSV Export**: All attendance data is stored in CSV files by date

## Requirements

- Python 3.8+
- Webcam for face capture and recognition
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages
   ```
   pip install -r requirements.txt
   ```

## Important Notes

- **Face Recognition**: This project uses the `face_recognition` library, which depends on `dlib`. Installation might be challenging on some systems. If you encounter issues:
  - On Windows: Ensure you have Visual C++ Build Tools installed
  - On macOS: You might need to install CMake (`brew install cmake`)
  - Consider using pre-built wheels if available for your platform

- **OpenCV**: Make sure your webcam is properly configured and accessible by OpenCV

## Usage

1. Start the application:
   ```
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Use the following features:
   - Register new students with their details
   - Capture student faces for recognition
   - Mark attendance through the webcam
   - View attendance records by date

## Folder Structure

- `app/templates/`: HTML templates for the web interface
- `app/static/`: Static assets (CSS, JavaScript)
- `data/faces/`: Stored face encodings for registered students
- `attendance csv/`: CSV files with attendance records (one file per day)

## License

This project is available for educational and personal use.

## Acknowledgements

- Built with Flask web framework
- Face recognition powered by face_recognition library
- Front-end using Bootstrap 5 