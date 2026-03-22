"""
Face encoding service using DeepFace for face detection and recognition.
DeepFace is easier to install and doesn't require compilation.
"""
import numpy as np
import cv2
from typing import Optional, List, Tuple
from deepface import DeepFace
from app.core.config import settings
import os
import tempfile


class FaceEncoder:
    """Service for encoding faces using DeepFace."""
    
    def __init__(self):
        """Initialize the face recognition model."""
        try:
            # DeepFace doesn't need explicit initialization
            # It will download models on first use
            # Use VGG-Face model (lightweight and accurate)
            self.model_name = "VGG-Face"  # Can also use "Facenet", "OpenFace", etc.
            self.initialized = True
            self._model_preloaded = False
            print("Face recognition service initialized successfully (DeepFace)")
        except Exception as e:
            print(f"Warning: Could not initialize DeepFace: {e}")
            self.initialized = False
    
    def l2_normalize(self, x: np.ndarray) -> np.ndarray:
        """L2 normalize a vector."""
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    
    def preload_model(self):
        """Preload the DeepFace model to avoid repeated downloads."""
        if self._model_preloaded:
            return True
            
        try:
            print("Preloading DeepFace model (this may take a moment on first run)...")
            # Create a dummy image to trigger model download
            dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, dummy_img)
                tmp_path = tmp_file.name
            
            try:
                # This will trigger model download
                DeepFace.represent(
                    img_path=tmp_path,
                    model_name=self.model_name,
                    enforce_detection=False,  # Don't fail on dummy image
                    detector_backend='opencv'
                )
                self._model_preloaded = True
                print("✅ DeepFace model preloaded successfully!")
                return True
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            print(f"Warning: Could not preload model: {e}")
            return False
    
    def encode_image(self, file_bytes: bytes) -> Optional[np.ndarray]:
        """
        Encode a face from image bytes.
        
        Args:
            file_bytes: Image file bytes
            
        Returns:
            Normalized face embedding or None if no face detected
        """
        if not self.initialized:
            return None
            
        try:
            # Save bytes to temporary file (DeepFace works with file paths)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Get face embedding using DeepFace
                embedding_obj = DeepFace.represent(
                    img_path=tmp_path,
                    model_name=self.model_name,
                    enforce_detection=True,
                    detector_backend='opencv'  # Use opencv for detection
                )
                
                if not embedding_obj or len(embedding_obj) == 0:
                    print("⚠️ No face detected in uploaded image")
                    return None
                
                # Extract embedding (first face)
                embedding = np.array(embedding_obj[0]['embedding'])
                return self.l2_normalize(embedding)
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None
    
    def detect_and_encode_faces(self, img: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Detect and encode all faces in an image.
        
        Args:
            img: OpenCV image (numpy array in BGR format)
            
        Returns:
            List of tuples (embedding, bbox) where bbox is (x1, y1, x2, y2)
        """
        if not self.initialized:
            return []
        
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                cv2.imwrite(tmp_file.name, img)
                tmp_path = tmp_file.name
            
            try:
                # Get all face embeddings and locations
                embedding_objs = DeepFace.represent(
                    img_path=tmp_path,
                    model_name=self.model_name,
                    enforce_detection=False,  # Don't fail if no face
                    detector_backend='opencv'
                )
                
                if not embedding_objs:
                    return []
                
                results = []
                
                for obj in embedding_objs:
                    embedding = np.array(obj['embedding'])
                    normalized_embedding = self.l2_normalize(embedding)
                    
                    # Get face region coordinates
                    # DeepFace returns 'facial_area' with x, y, w, h
                    facial_area = obj.get('facial_area', {})
                    x = facial_area.get('x', 0)
                    y = facial_area.get('y', 0)
                    w = facial_area.get('w', 0)
                    h = facial_area.get('h', 0)
                    
                    # Convert to (x1, y1, x2, y2) format
                    bbox = (x, y, x + w, y + h)
                    results.append((normalized_embedding, bbox))
                
                return results
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
