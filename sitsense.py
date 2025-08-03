#!/usr/bin/env python3
"""
SitSense - A Minimalist Posture Detection App
Nothing OS Inspired Design with Real-time Posture Monitoring
"""

import cv2
import customtkinter as ctk
import numpy as np
import threading
import time
import json
from PIL import Image, ImageTk
from plyer import notification
import os
import sys

# Try to import MediaPipe, fallback to OpenCV-only detection if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe loaded - Full posture detection available")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe not available - Using basic posture detection")

# Configure customtkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class PostureDetector:
    """Handles posture detection using MediaPipe or fallback OpenCV detection"""
    
    def __init__(self):
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.detection_method = "mediapipe"
        else:
            # Fallback to basic OpenCV detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detection_method = "opencv"
            
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def detect_posture_opencv(self, frame):
        """Basic posture detection using OpenCV face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        posture_status = "Good Posture"
        confidence = 0.8  # Default confidence for basic detection
        
        if len(faces) > 0:
            # Get the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Simple posture analysis based on face position
            frame_height, frame_width = frame.shape[:2]
            face_center_y = y + h // 2
            face_center_x = x + w // 2
            
            # Check if face is too close (face takes up too much of frame)
            face_area_ratio = (w * h) / (frame_width * frame_height)
            
            # Check if head is tilted (face not centered horizontally)
            horizontal_center_offset = abs(face_center_x - frame_width // 2) / frame_width
            
            # Simple posture rules
            if face_area_ratio > 0.15:  # Face too close
                posture_status = "Too Close to Screen"
            elif horizontal_center_offset > 0.3:  # Head tilted significantly
                posture_status = "Head Tilted"
            elif face_center_y < frame_height * 0.3:  # Head too high (slouching)
                posture_status = "Poor Posture - Slouching"
            
            # Add status text
            cv2.putText(frame, f"Detection: OpenCV", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Face Area: {face_area_ratio:.2f}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            posture_status = "No Face Detected"
            confidence = 0.0
            cv2.putText(frame, "No Face Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, posture_status, confidence
    
    def detect_posture_mediapipe(self, frame):
        """Full posture detection using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        posture_status = "Good Posture"
        confidence = 0.0
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get key points for posture analysis
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            left_ear = [landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y]
            right_ear = [landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value].y]
            nose = [landmarks[self.mp_pose.PoseLandmark.NOSE.value].x,
                   landmarks[self.mp_pose.PoseLandmark.NOSE.value].y]
            
            # Calculate neck angle (head forward posture)
            neck_angle = self.calculate_angle(
                [(left_shoulder[0] + right_shoulder[0])/2, (left_shoulder[1] + right_shoulder[1])/2],
                [(left_ear[0] + right_ear[0])/2, (left_ear[1] + right_ear[1])/2],
                nose
            )
            
            # Calculate shoulder slope (slouching detection)
            shoulder_slope = abs(left_shoulder[1] - right_shoulder[1])
            
            # Detect forward head posture
            head_forward = nose[1] < (left_ear[1] + right_ear[1])/2 - 0.02
            
            # Determine posture status
            confidence = min(results.pose_landmarks.landmark[0].visibility, 1.0)
            
            if head_forward or neck_angle < 160 or shoulder_slope > 0.05:
                posture_status = "Poor Posture"
            
            # Draw landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        return frame, posture_status, confidence
    
    def detect_posture(self, frame):
        """Detect posture from frame and return status"""
        if self.detection_method == "mediapipe":
            return self.detect_posture_mediapipe(frame)
        else:
            return self.detect_posture_opencv(frame)

class SitSenseApp:
    """Main application class with Nothing OS inspired UI"""
    
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("SitSense")
        
        # Load configuration
        self.config = self.load_config()
        
        self.root.geometry(f"{self.config['ui_settings']['window_width']}x{self.config['ui_settings']['window_height']}")
        self.root.configure(fg_color="#000000")
        
        # Initialize components
        self.posture_detector = PostureDetector()
        self.cap = None
        self.is_running = False
        self.last_alert_time = 0
        self.alert_cooldown = self.config['alert_cooldown']
        self.monitoring_thread = None
        
        self.setup_ui()
        self.setup_camera()
    
    def load_config(self):
        """Load configuration from config.json"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration if file doesn't exist
            return {
                "alert_cooldown": 30,
                "detection_confidence": 0.5,
                "tracking_confidence": 0.5,
                "posture_thresholds": {
                    "neck_angle_min": 160,
                    "shoulder_slope_max": 0.05,
                    "head_forward_threshold": 0.02
                },
                "ui_settings": {
                    "window_width": 600,
                    "window_height": 700,
                    "camera_width": 460,
                    "camera_height": 340
                },
                "camera_settings": {
                    "camera_index": 0,
                    "frame_width": 640,
                    "frame_height": 480,
                    "fps": 30
                },
                "notifications": {
                    "enabled": True,
                    "title": "SitSense - Posture Alert",
                    "message": "Poor posture detected! Please adjust your position.",
                    "timeout": 3
                }
            }
    
    def save_config(self):
        """Save current configuration to config.json"""
        try:
            with open('config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
        
    def setup_ui(self):
        """Setup the Nothing OS inspired UI"""
        # Main container
        self.main_frame = ctk.CTkFrame(self.root, fg_color="#000000", corner_radius=0)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="SitSense",
            font=ctk.CTkFont(family="SF Pro Display", size=32, weight="bold"),
            text_color="#FFFFFF"
        )
        self.title_label.pack(pady=(0, 20))
        
        # Webcam frame with rounded corners
        ui_config = self.config['ui_settings']
        self.camera_frame = ctk.CTkFrame(
            self.main_frame,
            width=ui_config['camera_width'],
            height=ui_config['camera_height'],
            fg_color="#111111",
            corner_radius=15,
            border_width=1,
            border_color="#333333"
        )
        self.camera_frame.pack(pady=(0, 20))
        self.camera_frame.pack_propagate(False)
        
        # Camera label
        self.camera_label = ctk.CTkLabel(
            self.camera_frame,
            text="Initializing Camera...",
            font=ctk.CTkFont(family="SF Pro Display", size=14),
            text_color="#FFFFFF"
        )
        self.camera_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Status panel
        self.status_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color="#111111",
            corner_radius=15,
            border_width=1,
            border_color="#333333"
        )
        self.status_frame.pack(fill="x", pady=(0, 20))
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame,
            text="Status: Initializing...",
            font=ctk.CTkFont(family="SF Pro Display", size=18, weight="bold"),
            text_color="#FFFFFF"
        )
        self.status_label.pack(pady=15)
        
        # Confidence indicator
        self.confidence_label = ctk.CTkLabel(
            self.status_frame,
            text="Detection Confidence: 0%",
            font=ctk.CTkFont(family="SF Pro Display", size=12),
            text_color="#888888"
        )
        self.confidence_label.pack(pady=(0, 15))
        
        # Control buttons
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(fill="x")
        
        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="Start Monitoring",
            command=self.toggle_monitoring,
            font=ctk.CTkFont(family="SF Pro Display", size=14, weight="bold"),
            fg_color="#FFFFFF",
            text_color="#000000",
            hover_color="#CCCCCC",
            corner_radius=25,
            height=40
        )
        self.start_button.pack(side="left", expand=True, fill="x", padx=(0, 10))
        
        self.settings_button = ctk.CTkButton(
            self.button_frame,
            text="Settings",
            command=self.show_settings,
            font=ctk.CTkFont(family="SF Pro Display", size=14),
            fg_color="transparent",
            text_color="#FFFFFF",
            hover_color="#222222",
            border_width=1,
            border_color="#333333",
            corner_radius=25,
            height=40
        )
        self.settings_button.pack(side="right", expand=True, fill="x", padx=(10, 0))
        
    def setup_camera(self):
        """Initialize camera"""
        try:
            camera_index = self.config['camera_settings']['camera_index']
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera {camera_index}")
            
            # Set camera properties from config
            camera_config = self.config['camera_settings']
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['frame_width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['frame_height'])
            self.cap.set(cv2.CAP_PROP_FPS, camera_config['fps'])
            
            self.camera_label.configure(text="Camera Ready")
            
        except (ValueError, cv2.error) as e:
            self.camera_label.configure(text=f"Camera Error: {str(e)}")
            print(f"Camera initialization error: {e}")
    
    def toggle_monitoring(self):
        """Start/stop posture monitoring"""
        if not self.is_running:
            if self.cap and self.cap.isOpened():
                self.is_running = True
                self.start_button.configure(text="Stop Monitoring", fg_color="#FF4444", hover_color="#CC3333")
                self.monitoring_thread = threading.Thread(target=self.monitoring_loop, daemon=True)
                self.monitoring_thread.start()
            else:
                self.show_notification("Camera not available", "Please check your camera connection.")
        else:
            self.is_running = False
            self.start_button.configure(text="Start Monitoring", fg_color="#FFFFFF", hover_color="#CCCCCC")
    
    def monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect posture
            processed_frame, posture_status, confidence = self.posture_detector.detect_posture(frame)
            
            # Convert frame for display
            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            ui_config = self.config['ui_settings']
            display_frame = cv2.resize(display_frame, (ui_config['camera_width'] - 20, ui_config['camera_height'] - 20))
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(display_frame)
            photo = ImageTk.PhotoImage(image=pil_image)
            
            # Update UI in main thread
            self.root.after(0, self.update_ui, photo, posture_status, confidence)
            
            # Check for poor posture alert
            if posture_status == "Poor Posture":
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_cooldown:
                    self.root.after(0, self.show_posture_alert)
                    self.last_alert_time = current_time
            
            time.sleep(0.033)  # ~30 FPS
    
    def update_ui(self, photo, posture_status, confidence):
        """Update UI elements"""
        # Update camera display
        self.camera_label.configure(image=photo, text="")
        self.camera_label.image = photo  # Keep a reference
        
        # Update status
        status_color = "#00FF00" if posture_status == "Good Posture" else "#FF4444"
        self.status_label.configure(
            text=f"Status: {posture_status}",
            text_color=status_color
        )
        
        # Update confidence
        self.confidence_label.configure(
            text=f"Detection Confidence: {int(confidence * 100)}%"
        )
    
    def show_posture_alert(self):
        """Show posture alert notification"""
        try:
            notification.notify(
                title="SitSense - Posture Alert",
                message="Poor posture detected! Please adjust your position.",
                timeout=3
            )
        except:
            # Fallback: show in-app alert
            alert_window = ctk.CTkToplevel(self.root)
            alert_window.title("Posture Alert")
            alert_window.geometry("300x150")
            alert_window.configure(fg_color="#000000")
            
            alert_label = ctk.CTkLabel(
                alert_window,
                text="Poor posture detected!\nPlease adjust your position.",
                font=ctk.CTkFont(family="SF Pro Display", size=14),
                text_color="#FFFFFF"
            )
            alert_label.pack(expand=True)
            
            # Auto-close after 3 seconds
            alert_window.after(3000, alert_window.destroy)
    
    def show_notification(self, title, message):
        """Show system notification"""
        try:
            notification.notify(
                title=f"SitSense - {title}",
                message=message,
                timeout=3
            )
        except:
            print(f"{title}: {message}")
    
    def show_settings(self):
        """Show settings window"""
        settings_window = ctk.CTkToplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.configure(fg_color="#000000")
        
        # Settings title
        title_label = ctk.CTkLabel(
            settings_window,
            text="Settings",
            font=ctk.CTkFont(family="SF Pro Display", size=24, weight="bold"),
            text_color="#FFFFFF"
        )
        title_label.pack(pady=20)
        
        # Alert cooldown setting
        cooldown_frame = ctk.CTkFrame(settings_window, fg_color="#111111", corner_radius=10)
        cooldown_frame.pack(fill="x", padx=20, pady=10)
        
        cooldown_label = ctk.CTkLabel(
            cooldown_frame,
            text="Alert Cooldown (seconds):",
            font=ctk.CTkFont(family="SF Pro Display", size=14),
            text_color="#FFFFFF"
        )
        cooldown_label.pack(pady=(15, 5))
        
        cooldown_var = ctk.StringVar(value=str(self.alert_cooldown))
        cooldown_entry = ctk.CTkEntry(
            cooldown_frame,
            textvariable=cooldown_var,
            font=ctk.CTkFont(family="SF Pro Display", size=12),
            width=100
        )
        cooldown_entry.pack(pady=(0, 15))
        
        # Save button
        def save_settings():
            try:
                new_cooldown = int(cooldown_var.get())
                self.alert_cooldown = new_cooldown
                self.config['alert_cooldown'] = new_cooldown
                self.save_config()
                settings_window.destroy()
                self.show_notification("Settings", "Settings saved successfully!")
            except ValueError:
                self.show_notification("Error", "Please enter a valid number for cooldown.")
        
        save_button = ctk.CTkButton(
            settings_window,
            text="Save Settings",
            command=save_settings,
            font=ctk.CTkFont(family="SF Pro Display", size=14, weight="bold"),
            fg_color="#FFFFFF",
            text_color="#000000",
            hover_color="#CCCCCC",
            corner_radius=25,
            height=40
        )
        save_button.pack(pady=20)
    
    def on_closing(self):
        """Handle application closing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = SitSenseApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()
