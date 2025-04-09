import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
import os
from PIL import Image
import time
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

class DeepfakeDetector:
    def __init__(self):
        # Set device for face detection
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device for face detection: {self.device}")
        
        # Initialize face detector
        self.mtcnn = MTCNN(
            keep_all=True,
            device=self.device,
            selection_method='probability',
            min_face_size=50
        )
        
        # Load the CNN model for deepfake detection
        try:
            print("Loading CNN model for deepfake detection...")
            
            # Look for the model in the following locations:
            # 1. In the DeepFake-Detector directory (as shown in your folder structure)
            # 2. In the current directory
            # 3. In the ml_models directory
            possible_model_paths = [
                os.path.join("D:", "DeepFake-Detector", "cnn_model.h5"),  # Your specific path
                os.path.join(os.getcwd(), "DeepFake-Detector", "cnn_model.h5"),
                os.path.join(os.path.dirname(__file__), "cnn_model.h5"),
            ]
            
            model_path = None
            for path in possible_model_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"Found model at: {model_path}")
                    break
            
            if model_path is None:
                raise FileNotFoundError("Could not find cnn_model.h5 in any of the expected locations")
            
            # Load the model
            self.model = load_model(model_path)
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Falling back to simplified detection...")
            self.model = None
        
    def extract_faces(self, frame, margin=0.2):
        """Extract faces from a frame using MTCNN with specified margin"""
        # Convert BGR to RGB (OpenCV uses BGR, PyTorch uses RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(frame_rgb)
        
        faces = []
        if boxes is not None:
            for i, box in enumerate(boxes):
                if probs[i] < 0.9:  # Filter low confidence detections
                    continue
                    
                box = box.astype(int)
                # Extract face with margin
                x1, y1, x2, y2 = box
                # Add margin
                width, height = x2 - x1, y2 - y1
                x1 = max(0, x1 - int(width * margin))
                y1 = max(0, y1 - int(height * margin))
                x2 = min(frame.shape[1], x2 + int(width * margin))
                y2 = min(frame.shape[0], y2 + int(height * margin))
                
                if x2 > x1 and y2 > y1:  # Ensure valid box dimensions
                    face = frame_rgb[y1:y2, x1:x2]
                    # Convert to PIL Image
                    pil_face = Image.fromarray(face)
                    faces.append(pil_face)
                    
        return faces
    
    def analyze_face_quality(self, face_pil):
        """Analyze the quality of a face image to detect abnormalities (fallback method)"""
        # Convert PIL to numpy for analysis
        face_np = np.array(face_pil)
        
        # Check if the face image is valid
        if face_np.size == 0:
            return 0.5
            
        # Convert to grayscale for texture analysis
        if len(face_np.shape) == 3:
            gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = face_np
            
        # Analyze blur - blurry faces in real videos could indicate deepfakes
        # or excessive blur could be used to hide artifacts
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Check color consistency - unusual color patterns can indicate deepfakes
        hsv = cv2.cvtColor(face_np, cv2.COLOR_RGB2HSV) if len(face_np.shape) == 3 else None
        color_score = 0.5
        if hsv is not None:
            s = hsv[:,:,1]  # Saturation channel
            s_mean = np.mean(s)
            # Unusual saturation can indicate manipulation
            if s_mean < 20 or s_mean > 180:
                color_score = 0.7
                
        # Noise analysis - deepfakes often have different noise patterns
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float32) - blur.astype(np.float32)
        noise_std = np.std(noise)
        
        # Combine factors
        if laplacian_var < 100:  # Very blurry
            blur_score = 0.7  # Suspicious
        elif laplacian_var > 1000:  # Very sharp, possibly artificial
            blur_score = 0.6  # Somewhat suspicious
        else:
            blur_score = 0.3  # Likely normal
            
        # Weight the factors
        quality_score = (0.4 * blur_score) + (0.3 * color_score) + (0.3 * min(1.0, noise_std / 50))
        
        return quality_score
    
    def predict_single_face(self, face_pil):
        """Predict if a face is real or fake using the CNN model"""
        if self.model is None:
            # Fallback to simplified detection if model couldn't be loaded
            return self.analyze_face_quality(face_pil)
        
        try:
            # Resize to 128x128 as expected by the model (based on your documentation)
            face_resized = face_pil.resize((128, 128))
            
            # Convert to numpy array and normalize (0-1 range)
            face_array = keras_image.img_to_array(face_resized) / 255.0
            
            # Add batch dimension
            face_array = np.expand_dims(face_array, axis=0)
            
            # Get prediction
            prediction = self.model.predict(face_array, verbose=0)
            
            # Based on the documentation:
            # print('Real' if prediction[0][0] < 0.5 else 'Fake')
            # So if prediction[0][0] >= 0.5, it's a fake (1); otherwise it's real (0)
            # We need to return a probability between 0-1 where 1 means fake
            fake_probability = prediction[0][0]
            
            # The documentation example suggests prediction[0][0] < 0.5 is "Real"
            # which means prediction[0][0] >= 0.5 is "Fake"
            # So we can directly use this value as our fake probability
            
            return fake_probability
            
        except Exception as e:
            print(f"Error predicting face: {str(e)}")
            # Fall back to quality analysis
            return self.analyze_face_quality(face_pil)
    
    def predict(self, video_path):
        """Detect deepfake in a video using the CNN model"""
        print(f"Analyzing video: {video_path}")
        start_time = time.time()
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        print(f"Video has {frame_count} frames at {fps} fps, duration: {duration:.2f}s")
        
        # Skip frames for longer videos
        frame_skip = max(1, int(frame_count / 30)) if frame_count > 30 else 1
        
        # For analysis
        all_scores = []
        frames_processed = 0
        faces_analyzed = 0
        
        while cap.isOpened() and frames_processed < 50:  # Process max 50 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            if frames_processed % frame_skip == 0:
                # Extract faces from the frame
                faces = self.extract_faces(frame)
                faces_analyzed += len(faces)
                
                # Process each face
                for face_pil in faces:
                    # Get prediction
                    fake_score = self.predict_single_face(face_pil)
                    all_scores.append(fake_score)
                    print(f"Frame {frames_processed}, Face score: {fake_score:.4f}")
            
            frames_processed += 1
            
        cap.release()
        
        # Handle edge cases
        if not all_scores:
            print("No faces detected in the video or processing failed.")
            # Return a slightly higher score when no faces are detected (suspicious)
            return 0.6
            
        # Calculate final score
        # Multiple approaches to aggregate the scores:
        
        # 1. Mean of all scores
        mean_score = np.mean(all_scores)
        
        # 2. Take the maximum score (most likely fake face)
        max_score = np.max(all_scores)
        
        # 3. Count high-score faces (how many faces scored > 0.7)
        high_score_ratio = sum(1 for score in all_scores if score > 0.7) / len(all_scores) if all_scores else 0
        
        # Calculate final weighted score
        final_score = (0.5 * mean_score) + (0.3 * max_score) + (0.2 * high_score_ratio)
        
        # Adjust confidence based on number of faces analyzed
        if faces_analyzed < 3:
            # Not enough faces for high confidence - pull slightly toward middle
            final_score = (final_score * 0.7) + 0.15
        
        processing_time = time.time() - start_time
        print(f"Analysis complete in {processing_time:.2f} seconds")
        print(f"Processed {frames_processed} frames, analyzed {faces_analyzed} faces")
        print(f"Mean score: {mean_score:.4f}, Max score: {max_score:.4f}, Final score: {final_score:.4f}")
        
        return final_score