import cv2
import mediapipe as mp
import numpy as np
import time
from enum import Enum

class ExerciseType(Enum):
    PUSHUP = "Push-up"
    DUMBBELL = "Dumbbell Lift"
    BICEP_CURL = "Bicep Curl"
    TRICEP_EXTENSION = "Tricep Extension"

class ExerciseTracker:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # Exercise tracking variables
        self.counter = 0
        self.stage = None
        self.start_time = time.time()
        self.calories_per_minute = {
            ExerciseType.PUSHUP: 8,    # calories burned per minute
            ExerciseType.DUMBBELL: 6,
            ExerciseType.BICEP_CURL: 5,
            ExerciseType.TRICEP_EXTENSION: 5
        }
        
    def calculate_angle(self, a, b, c):
        a = np.array(a)  
        b = np.array(b)  
        c = np.array(c)  
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def process_frame(self, frame, exercise_type):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        # Convert back to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Get landmark coordinates
            landmarks = results.pose_landmarks.landmark
            
            # Extract relevant points based on exercise type
            if exercise_type == ExerciseType.PUSHUP:
                return self.track_pushup(image, landmarks)
            elif exercise_type == ExerciseType.DUMBBELL:
                return self.track_dumbbell(image, landmarks)
            elif exercise_type == ExerciseType.BICEP_CURL:
                return self.track_bicep_curl(image, landmarks)
            elif exercise_type == ExerciseType.TRICEP_EXTENSION:
                return self.track_tricep_extension(image, landmarks)
                
        return image
    
    def track_pushup(self, image, landmarks):
        """Track push-up movement and count repetitions"""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        # Calculate elbow angle
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Push-up counter logic
        if angle > 160:
            self.stage = "up"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            
        return self.update_display(image, ExerciseType.PUSHUP)
    
    def track_dumbbell(self, image, landmarks):
        """Track dumbbell lift movement and count repetitions"""
        # Get coordinates for shoulder, elbow, and wrist
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Calculate elbow angle
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Dumbbell lift counter logic
        if angle < 30:
            self.stage = "down"
        if angle > 150 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            
        return self.update_display(image, ExerciseType.DUMBBELL)
    
    def track_bicep_curl(self, image, landmarks):
        """Track bicep curl movement and count repetitions"""
        # Get coordinates for shoulder, elbow, and wrist
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Calculate elbow angle
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Bicep curl counter logic
        if angle > 160:
            self.stage = "down"
        if angle < 50 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            
        return self.update_display(image, ExerciseType.BICEP_CURL)
    
    def track_tricep_extension(self, image, landmarks):
        """Track tricep extension movement and count repetitions"""
        # Get coordinates for shoulder, elbow, and wrist
        shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        
        # Calculate elbow angle
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Tricep extension counter logic
        if angle < 30:
            self.stage = "up"
        if angle > 150 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            
        return self.update_display(image, ExerciseType.TRICEP_EXTENSION)
    
    def update_display(self, image, exercise_type):
        """Update the display with exercise stats"""
        # Display stats
        cv2.putText(image, f'Exercise: {exercise_type.value}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black font
        cv2.putText(image, f'Reps: {self.counter}', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black font
        
        if exercise_type == ExerciseType.DUMBBELL:
            total_weight = self.counter * 10  # Dumbbell weight is 10 kg
            cv2.putText(image, f'Total Weight: {total_weight} kg', 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black font
        
        cv2.putText(image, f'Stage: {self.stage if self.stage else "N/A"}', 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black font
        
        return image

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 for webcam
    
    # Let user choose exercise type
    print("Select exercise type:")
    print("1. Push-ups")
    print("2. Dumbbell Lifts")
    print("3. Bicep Curls")
    print("4. Tricep Extensions")
    choice = input("Enter choice (1-4): ")
    
    exercise_map = {
        "1": ExerciseType.PUSHUP,
        "2": ExerciseType.DUMBBELL,
        "3": ExerciseType.BICEP_CURL,
        "4": ExerciseType.TRICEP_EXTENSION
    }
    exercise_type = exercise_map.get(choice, ExerciseType.PUSHUP)
    tracker = ExerciseTracker()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame and track exercise
        frame = tracker.process_frame(frame, exercise_type)
        
        # Display the frame
        cv2.imshow('Exercise Tracker', frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    print(f"\nFinal Stats:")
    print(f"Exercise: {exercise_type.value}")
    print(f"Total Reps: {tracker.counter}")
    elapsed_time = (time.time() - tracker.start_time) / 60
    total_calories = elapsed_time * tracker.calories_per_minute[exercise_type]
    print(f"Total Calories Burned: {total_calories:.2f}")

if __name__ == "__main__":
    main()