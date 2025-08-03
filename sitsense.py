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
import math  # Added for proper angle calculations
from PIL import Image, ImageTk
from plyer import notification
import os
import sys

# Try to import CVZone, fallback to OpenCV-only detection if not available
try:
    from cvzone.PoseModule import PoseDetector
    CVZONE_AVAILABLE = True
    print("✅ CVZone loaded - Full posture detection available")
except ImportError:
    CVZONE_AVAILABLE = False
    print("⚠️  CVZone not available - Using basic posture detection")

# Configure customtkinter appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

class PostureDetector:
    """Handles posture detection using CVZone or fallback OpenCV detection"""
    
    def __init__(self, force_opencv=False):
        self.force_opencv = force_opencv
        self.issue_history = []  # Track issues over time for consistency
        self.history_length = 10  # Number of frames to consider
        
        if CVZONE_AVAILABLE and not force_opencv:
            self.detector = PoseDetector(staticMode=False, 
                                       modelComplexity=1, 
                                       smoothLandmarks=True, 
                                       enableSegmentation=False, 
                                       smoothSegmentation=True, 
                                       detectionCon=0.5, 
                                       trackCon=0.5)
            self.detection_method = "cvzone"
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
    
    def check_consistent_issues(self, current_issues):
        """Check if issues are consistent over multiple frames"""
        self.issue_history.append(current_issues)
        
        # Keep only recent history
        if len(self.issue_history) > self.history_length:
            self.issue_history.pop(0)
        
        # Need at least 5 frames of history
        if len(self.issue_history) < 5:
            return []
        
        # Count how often each issue appears
        issue_counts = {}
        for frame_issues in self.issue_history:
            for issue in frame_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Only return issues that appear in at least 60% of recent frames
        consistent_threshold = len(self.issue_history) * 0.6
        consistent_issues = [issue for issue, count in issue_counts.items() 
                           if count >= consistent_threshold]
        
        return consistent_issues
    
    def detect_posture_opencv(self, frame):
        """Enhanced posture detection using OpenCV face detection"""
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
            
            # Enhanced posture analysis based on face position
            frame_height, frame_width = frame.shape[:2]
            face_center_y = y + h // 2
            face_center_x = x + w // 2
            face_bottom = y + h
            face_top = y
            
            # Calculate various metrics
            face_area_ratio = (w * h) / (frame_width * frame_height)
            horizontal_center_offset = abs(face_center_x - frame_width // 2) / frame_width
            vertical_position = face_center_y / frame_height
            face_aspect_ratio = w / h if h > 0 else 1
            
            # Enhanced posture detection rules - balanced thresholds for basic mode
            issues = []
            
            # 1. Too close to screen (face takes up too much space) - moderate
            if face_area_ratio > 0.22:  # Balanced threshold
                issues.append("Too Close to Screen")
            
            # 2. Head tilted significantly (not centered horizontally) - moderate
            elif horizontal_center_offset > 0.25:  # More reasonable threshold
                issues.append("Head Tilted")
            
            # 3. Slouching detection (face too high in frame - head dropping) - moderate
            elif vertical_position < 0.20:  # Balanced threshold
                issues.append("Slouching - Head Down")
            
            # 4. Leaning back detection (face too low in frame) - moderate
            elif vertical_position > 0.75:  # Balanced threshold
                issues.append("Leaning Back")
            
            # 5. Head forward posture (face appears wider due to perspective) - moderate
            elif face_aspect_ratio > 1.4:  # More reasonable threshold
                issues.append("Head Forward")
            
            # 6. Check for extreme tilting based on face dimensions - moderate
            elif abs(w - h) > w * 0.4:  # More reasonable threshold
                issues.append("Head Tilted")
            
            # Check for consistent issues over time - balanced for basic mode
            consistent_issues = self.check_consistent_issues(issues)
            
            # Set posture status based on consistent issues - balanced approach
            if len(consistent_issues) >= 2:
                posture_status = f"Multiple Issues: {', '.join(consistent_issues[:2])}"
            elif len(consistent_issues) == 1:
                # Single consistent issue - flag most issues but be reasonable
                posture_status = f"Poor Posture - {consistent_issues[0]}"
            else:
                posture_status = "Good Posture"  # No consistent issues
            
            # Add detailed status overlay
            cv2.putText(frame, f"Detection: OpenCV Enhanced", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Face Area: {face_area_ratio:.2f}", (10, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"V-Pos: {vertical_position:.2f}", (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"H-Offset: {horizontal_center_offset:.2f}", (10, 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw reference lines for posture zones
            # Good posture zone (middle third of screen)
            good_zone_top = int(frame_height * 0.35)
            good_zone_bottom = int(frame_height * 0.65)
            cv2.line(frame, (0, good_zone_top), (frame_width, good_zone_top), (0, 255, 0), 1)
            cv2.line(frame, (0, good_zone_bottom), (frame_width, good_zone_bottom), (0, 255, 0), 1)
            cv2.putText(frame, "Good Zone", (frame_width - 100, good_zone_top + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Center line for head tilt reference
            center_x = frame_width // 2
            cv2.line(frame, (center_x, 0), (center_x, frame_height), (255, 255, 0), 1)
            
        else:
            posture_status = "No Face Detected"
            confidence = 0.0
            cv2.putText(frame, "No Face Detected - Sit in view", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame, posture_status, confidence
    
    def detect_posture_cvzone(self, frame):
        """Enhanced full body posture detection using CVZone with proper angle calculations"""
        frame = self.detector.findPose(frame, draw=True)
        lmList, bboxInfo = self.detector.findPosition(frame, draw=False)
        
        posture_status = "Good Posture"
        confidence = 0.8  # Default confidence for CVZone
        
        if lmList:
            # CVZone landmark positions - [x, y, z] format
            nose = lmList[0] if len(lmList) > 0 else None
            left_shoulder = lmList[11] if len(lmList) > 11 else None
            right_shoulder = lmList[12] if len(lmList) > 12 else None
            left_hip = lmList[23] if len(lmList) > 23 else None
            right_hip = lmList[24] if len(lmList) > 24 else None
            
            # Check if we have essential landmarks
            if all([nose, left_shoulder, right_shoulder]):
                h, w = frame.shape[:2]
                issues = []
                
                # 1. NECK ANGLE (Forward Head Tilt) - Using your formula
                mid_shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
                mid_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                
                dx = nose[0] - mid_shoulder_x
                dy = nose[1] - mid_shoulder_y
                neck_angle = math.degrees(math.atan2(dy, dx))
                
                # Normalize neck angle to 0-180 range for easier interpretation
                if neck_angle < 0:
                    neck_angle += 180
                
                # ✅ Correct posture: ~90° (head above shoulders)
                # ⚠️ Bad posture (slouching): < 70° or > 110°
                if neck_angle < 70 or neck_angle > 110:
                    issues.append("Forward Head Posture")
                
                # 2. SHOULDER TILT ANGLE (Side Lean) - Using your formula
                dx_shoulder = right_shoulder[0] - left_shoulder[0]
                dy_shoulder = right_shoulder[1] - left_shoulder[1]
                shoulder_tilt = math.degrees(math.atan2(dy_shoulder, dx_shoulder))
                
                # ✅ Correct: angle ~0° (horizontal)
                # ⚠️ Bad posture: > ±10°
                if abs(shoulder_tilt) > 10:
                    if shoulder_tilt > 0:
                        issues.append("Leaning Right")
                    else:
                        issues.append("Leaning Left")
                
                # 3. BACK/SPINE LEAN (if hips are available) - Using your formula
                if left_hip:
                    dx_spine = left_shoulder[0] - left_hip[0]
                    dy_spine = left_shoulder[1] - left_hip[1]
                    spine_angle = math.degrees(math.atan2(dy_spine, dx_spine))
                    
                    # Normalize spine angle
                    if spine_angle < 0:
                        spine_angle += 180
                    
                    # ✅ Upright: 90° (vertical alignment)
                    # ⚠️ Leaning/slouching: < 75° or > 105°
                    if spine_angle < 75:
                        issues.append("Leaning Forward")
                    elif spine_angle > 105:
                        issues.append("Leaning Backward")
                
                # 4. DISTANCE TO CAMERA (Depth Detection) - Using your formula
                z_nose = nose[2] if len(nose) > 2 else 0
                z_shoulders = (left_shoulder[2] + right_shoulder[2]) / 2 if len(left_shoulder) > 2 and len(right_shoulder) > 2 else 0
                depth_diff = z_nose - z_shoulders
                
                # ✅ Proper distance: depth_diff ~0–20
                # ⚠️ Leaning in too close: > 40
                if depth_diff > 40:
                    issues.append("Too Close to Camera")
                elif depth_diff < -20:  # Head behind shoulders significantly
                    issues.append("Head Too Far Back")
                
                # Check for consistent issues over time
                consistent_issues = self.check_consistent_issues(issues)
                
                # Set posture status based on consistent issues
                if len(consistent_issues) >= 2:
                    posture_status = f"Multiple Issues: {', '.join(consistent_issues[:2])}"
                elif len(consistent_issues) == 1:
                    # Flag significant issues immediately
                    if consistent_issues[0] in ["Forward Head Posture", "Too Close to Camera", "Leaning Forward"]:
                        posture_status = f"Poor Posture - {consistent_issues[0]}"
                    else:
                        # Minor single issues - be more lenient
                        posture_status = f"Minor Issue - {consistent_issues[0]}"
                else:
                    posture_status = "Good Posture"
                
                # Draw analysis information on frame
                cv2.putText(frame, f"Detection: CVZone - Angle Based", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}° (90° ideal)", (10, h-120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Shoulder Tilt: {shoulder_tilt:.1f}° (0° ideal)", (10, h-100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if left_hip:
                    cv2.putText(frame, f"Spine Angle: {spine_angle:.1f}° (90° ideal)", (10, h-80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Depth Diff: {depth_diff:.1f} (0-20 ideal)", (10, h-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Issues: {len(issues)}", (10, h-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Draw shoulder alignment line with color coding
                shoulder_color = (0, 255, 0) if abs(shoulder_tilt) <= 10 else (0, 0, 255)
                cv2.line(frame, 
                        (int(left_shoulder[0]), int(left_shoulder[1])),
                        (int(right_shoulder[0]), int(right_shoulder[1])),
                        shoulder_color, 3)
                
                # Draw neck alignment line with color coding
                neck_color = (0, 255, 0) if 70 <= neck_angle <= 110 else (0, 0, 255)
                cv2.line(frame,
                        (int(mid_shoulder_x), int(mid_shoulder_y)),
                        (int(nose[0]), int(nose[1])),
                        neck_color, 2)
                
                # Draw spine alignment line if hip is available
                if left_hip:
                    spine_color = (0, 255, 0) if 75 <= spine_angle <= 105 else (0, 0, 255)
                    cv2.line(frame,
                            (int(left_hip[0]), int(left_hip[1])),
                            (int(left_shoulder[0]), int(left_shoulder[1])),
                            spine_color, 2)
                
                # Distance indicator (circle at nose)
                distance_color = (0, 255, 0) if 0 <= depth_diff <= 20 else (0, 0, 255)
                cv2.circle(frame, (int(nose[0]), int(nose[1])), 8, distance_color, -1)
                
                # Add posture status indicators
                if neck_angle < 70 or neck_angle > 110:
                    cv2.putText(frame, "HEAD POSTURE!", (w-200, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif abs(shoulder_tilt) > 10:
                    cv2.putText(frame, "TILTED!", (w-200, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                elif depth_diff > 40:
                    cv2.putText(frame, "TOO CLOSE!", (w-200, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                elif left_hip and (spine_angle < 75 or spine_angle > 105):
                    cv2.putText(frame, "SPINE LEAN!", (w-200, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 50, 255), 2)
                else:
                    cv2.putText(frame, "GOOD POSTURE", (w-200, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Add angle indicators with ideal ranges
                cv2.putText(frame, f"Ideal Ranges:", (w-200, h-120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Neck: 70-110°", (w-200, h-100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Shoulder: ±10°", (w-200, h-80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Spine: 75-105°", (w-200, h-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(frame, f"Depth: 0-20", (w-200, h-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
        else:
            # No pose detected
            cv2.putText(frame, "No Pose Detected - Sit in view", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            posture_status = "No Pose Detected"
            confidence = 0.0
        
        return frame, posture_status, confidence
    
    def detect_posture(self, frame):
        """Detect posture from frame and return status"""
        if self.detection_method == "cvzone":
            return self.detect_posture_cvzone(frame)
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
        
        # Detection mode setting
        self.detection_mode = self.config.get('detection_mode', 'full')  # 'basic' or 'full'
        
        # Initialize components
        force_opencv = (self.detection_mode == 'basic')
        self.posture_detector = PostureDetector(force_opencv=force_opencv)
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
                "alert_cooldown": 30,  # Reasonable alert interval
                "detection_confidence": 0.5,
                "tracking_confidence": 0.5,
                "detection_mode": "full",  # 'basic' or 'full'
                "posture_thresholds": {
                    # CVZone angle-based thresholds (from angle calculations)
                    "neck_angle_min": 70,  # Minimum neck angle (degrees)
                    "neck_angle_max": 110,  # Maximum neck angle (degrees)
                    "shoulder_tilt_max": 10,  # Maximum shoulder tilt (degrees)
                    "spine_angle_min": 75,  # Minimum spine angle (degrees)  
                    "spine_angle_max": 105,  # Maximum spine angle (degrees)
                    "depth_diff_min": 0,  # Minimum depth difference
                    "depth_diff_max": 20,  # Maximum depth difference
                    "depth_diff_close_threshold": 40,  # Too close threshold
                    # Basic mode thresholds
                    "face_area_ratio_max": 0.22,
                    "horizontal_offset_max": 0.25,
                    "vertical_position_min": 0.20,
                    "vertical_position_max": 0.75
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
        title_text = "SitSense"
        if self.detection_mode == 'basic' or not CVZONE_AVAILABLE:
            title_text += " (Basic Mode)"
        else:
            title_text += " (Full Mode)"
        
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text=title_text,
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
            
            # Check for poor posture alert (angle-based CVZone conditions)
            poor_posture_conditions = [
                "Poor Posture" in posture_status,
                "Too Close" in posture_status,
                "Slouching" in posture_status,
                "Leaning" in posture_status,
                "Forward" in posture_status,
                "Tilted" in posture_status,
                "Multiple Issues" in posture_status,
                "Hunched" in posture_status,
                # CVZone angle-based specific conditions
                "Forward Head Posture" in posture_status,
                "Leaning Right" in posture_status,
                "Leaning Left" in posture_status,
                "Leaning Forward" in posture_status,
                "Leaning Backward" in posture_status,
                "Too Close to Camera" in posture_status,
                "Head Too Far Back" in posture_status
            ]
            
            if any(poor_posture_conditions):
                current_time = time.time()
                if current_time - self.last_alert_time > self.alert_cooldown:
                    self.root.after(0, self.show_posture_alert, posture_status)
                    self.last_alert_time = current_time
            
            time.sleep(0.033)  # ~30 FPS
    
    def update_ui(self, photo, posture_status, confidence):
        """Update UI elements with enhanced status colors"""
        # Update camera display
        self.camera_label.configure(image=photo, text="")
        self.camera_label.image = photo  # Keep a reference
        
        # Enhanced status color coding for angle-based detection
        if posture_status == "Good Posture":
            status_color = "#00FF00"  # Green
        elif "Multiple Issues" in posture_status:
            status_color = "#FF0000"  # Red for multiple issues
        elif "Minor Issue" in posture_status:
            status_color = "#FFA500"  # Orange for minor issues
        elif any(issue in posture_status for issue in ["Too Close", "Forward Head Posture", "Leaning Forward"]):
            status_color = "#FF8800"  # Orange for moderate issues
        elif any(issue in posture_status for issue in ["Leaning Right", "Leaning Left", "Leaning Backward", "Head Too Far Back"]):
            status_color = "#FF4444"  # Red for serious posture issues
        elif any(issue in posture_status for issue in ["Slouching", "Tilted", "Hunched"]):
            status_color = "#FF4444"  # Red for serious posture issues
        elif "No Face Detected" in posture_status or "No Pose Detected" in posture_status:
            status_color = "#888888"  # Gray for no detection
        else:
            status_color = "#FFFF00"  # Yellow for other issues
        
        self.status_label.configure(
            text=f"Status: {posture_status}",
            text_color=status_color
        )
        
        # Update confidence with color coding
        if confidence > 0.7:
            conf_color = "#00FF00"  # Green for high confidence
        elif confidence > 0.4:
            conf_color = "#FFFF00"  # Yellow for medium confidence
        else:
            conf_color = "#FF4444"  # Red for low confidence
            
        self.confidence_label.configure(
            text=f"Detection Confidence: {int(confidence * 100)}%",
            text_color=conf_color
        )
    
    def show_posture_alert(self, posture_status="Poor posture detected"):
        """Show posture alert notification with specific issue"""
        try:
            # Create specific message based on posture issue
            if "Too Close" in posture_status:
                message = "You're sitting too close to the screen! Move back a bit."
            elif "Forward Head Posture" in posture_status:
                message = "Forward head posture detected! Pull your head back over your shoulders."
            elif "Leaning Right" in posture_status:
                message = "You're leaning to the right! Straighten your shoulders."
            elif "Leaning Left" in posture_status:
                message = "You're leaning to the left! Straighten your shoulders."
            elif "Leaning Forward" in posture_status:
                message = "You're leaning forward! Sit up straight and align your spine."
            elif "Leaning Backward" in posture_status:
                message = "You're leaning too far back! Sit upright."
            elif "Head Too Far Back" in posture_status:
                message = "Your head is too far back! Bring it forward slightly."
            elif "Slouching" in posture_status or "Head Down" in posture_status:
                message = "You're slouching! Sit up straight and lift your head."
            elif "Forward" in posture_status:
                message = "Your head is too far forward! Pull your head back."
            elif "Tilted" in posture_status:
                message = "Your head is tilted! Straighten your head position."
            elif "Hunched" in posture_status:
                message = "You're hunched forward! Straighten your back."
            elif "Multiple Issues" in posture_status:
                message = "Multiple posture issues detected! Please adjust your position."
            else:
                message = "Poor posture detected! Please adjust your position."
            
            notification.notify(
                title="SitSense - Posture Alert",
                message=message,
                timeout=4
            )
        except:
            # Fallback: show in-app alert
            alert_window = ctk.CTkToplevel(self.root)
            alert_window.title("Posture Alert")
            alert_window.geometry("350x180")
            alert_window.configure(fg_color="#000000")
            
            # Make sure window appears on top
            alert_window.attributes('-topmost', True)
            
            alert_label = ctk.CTkLabel(
                alert_window,
                text=f"⚠️ {posture_status}\n\nPlease adjust your position!",
                font=ctk.CTkFont(family="SF Pro Display", size=14),
                text_color="#FFFFFF",
                justify="center"
            )
            alert_label.pack(expand=True, pady=20)
            
            # Dismiss button
            dismiss_button = ctk.CTkButton(
                alert_window,
                text="Got it!",
                command=alert_window.destroy,
                font=ctk.CTkFont(family="SF Pro Display", size=12),
                fg_color="#FFFFFF",
                text_color="#000000",
                hover_color="#CCCCCC",
                corner_radius=15,
                width=80,
                height=30
            )
            dismiss_button.pack(pady=(0, 20))
            
            # Auto-close after 5 seconds
            alert_window.after(5000, alert_window.destroy)
    
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
        settings_window.geometry("400x400")  # Made taller for more options
        settings_window.configure(fg_color="#000000")
        
        # Settings title
        title_label = ctk.CTkLabel(
            settings_window,
            text="Settings",
            font=ctk.CTkFont(family="SF Pro Display", size=24, weight="bold"),
            text_color="#FFFFFF"
        )
        title_label.pack(pady=20)
        
        # Detection mode setting
        mode_frame = ctk.CTkFrame(settings_window, fg_color="#111111", corner_radius=10)
        mode_frame.pack(fill="x", padx=20, pady=10)
        
        mode_label = ctk.CTkLabel(
            mode_frame,
            text="Detection Mode:",
            font=ctk.CTkFont(family="SF Pro Display", size=14),
            text_color="#FFFFFF"
        )
        mode_label.pack(pady=(15, 5))
        
        mode_var = ctk.StringVar(value=self.detection_mode)
        mode_option = ctk.CTkOptionMenu(
            mode_frame,
            values=["basic", "full"],
            variable=mode_var,
            font=ctk.CTkFont(family="SF Pro Display", size=12),
            width=150
        )
        mode_option.pack(pady=(0, 10))
        
        # Add mode descriptions
        mode_desc = ctk.CTkLabel(
            mode_frame,
            text="Basic: Face detection only (less strict)\nFull: Complete body posture analysis",
            font=ctk.CTkFont(family="SF Pro Display", size=10),
            text_color="#AAAAAA",
            justify="left"
        )
        mode_desc.pack(pady=(0, 15))
        
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
                new_mode = mode_var.get()
                
                # Update settings
                self.alert_cooldown = new_cooldown
                self.detection_mode = new_mode
                self.config['alert_cooldown'] = new_cooldown
                self.config['detection_mode'] = new_mode
                self.save_config()
                
                # Recreate posture detector with new mode
                force_opencv = (new_mode == 'basic')
                self.posture_detector = PostureDetector(force_opencv=force_opencv)
                
                # Update UI title
                title_text = "SitSense"
                if new_mode == 'basic' or not CVZONE_AVAILABLE:
                    title_text += " (Basic Mode)"
                else:
                    title_text += " (Full Mode)"
                self.title_label.configure(text=title_text)
                
                settings_window.destroy()
                self.show_notification("Settings", "Settings saved successfully! Changes will take effect immediately.")
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
