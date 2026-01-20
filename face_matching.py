"""
Face Matching Module for KYC Verification

Compares the live selfie captured during liveness detection
with the face extracted from the ID document.

Uses face embeddings for comparison (simulated for demo,
can be replaced with actual models like FaceNet, ArcFace, etc.)
"""

import logging
import base64
import io
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Try to import face detection libraries
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available for face detection")

# Face detection cascade (built into OpenCV)
face_cascade = None
if CV2_AVAILABLE:
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        logger.info("Face detection cascade loaded")
    except Exception as e:
        logger.warning(f"Could not load face cascade: {e}")


def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """Decode a base64 image string to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return np.array(image.convert('RGB'))
    except Exception as e:
        logger.error(f"Error decoding base64 image: {e}")
        return None


def detect_face(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face in image and return bounding box.
    Returns (x, y, width, height) or None if no face found.
    """
    if not CV2_AVAILABLE or face_cascade is None:
        # Simulated face detection for demo
        h, w = image.shape[:2]
        # Return center region as "face"
        face_w, face_h = int(w * 0.4), int(h * 0.5)
        x = (w - face_w) // 2
        y = (h - face_h) // 3
        return (x, y, face_w, face_h)
    
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50)
        )
        
        if len(faces) > 0:
            # Return largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            return tuple(faces[0])
        return None
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return None


def extract_face_region(image: np.ndarray, bbox: Tuple[int, int, int, int], padding: float = 0.2) -> np.ndarray:
    """Extract face region from image with padding."""
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    
    # Add padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)
    
    return image[y1:y2, x1:x2]


def compute_face_embedding(face_image: np.ndarray) -> np.ndarray:
    """
    Compute face embedding (feature vector).
    
    In production, use a proper face recognition model like:
    - FaceNet
    - ArcFace
    - DeepFace
    
    For demo, we use a simple histogram-based approach.
    """
    try:
        # Resize to standard size
        face_pil = Image.fromarray(face_image)
        face_pil = face_pil.resize((128, 128))
        face_np = np.array(face_pil)
        
        # Simple feature extraction (color histogram)
        # In production, replace with neural network embeddings
        embedding = []
        for channel in range(3):
            hist, _ = np.histogram(face_np[:, :, channel], bins=32, range=(0, 256))
            embedding.extend(hist / hist.sum())  # Normalize
        
        return np.array(embedding)
    except Exception as e:
        logger.error(f"Embedding extraction error: {e}")
        return np.zeros(96)


def compare_faces(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compare two face embeddings and return similarity score (0-100).
    
    Uses cosine similarity.
    """
    try:
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Convert to percentage (0-100)
        # Adjust threshold: cosine similarity of 0.7+ is usually considered a match
        score = max(0, min(100, (similarity - 0.3) / 0.7 * 100))
        
        return round(score, 1)
    except Exception as e:
        logger.error(f"Face comparison error: {e}")
        return 0.0


def match_faces(selfie_base64: str, document_image_path: str) -> Dict[str, Any]:
    """
    Match face from selfie with face on document.
    
    Args:
        selfie_base64: Base64 encoded selfie image
        document_image_path: Path to document image file
        
    Returns:
        Dictionary with match result, score, and details
    """
    result = {
        "success": False,
        "match": False,
        "score": 0.0,
        "message": "",
        "details": {}
    }
    
    try:
        # Decode selfie
        selfie_image = decode_base64_image(selfie_base64)
        if selfie_image is None:
            result["message"] = "Could not decode selfie image"
            return result
        
        # Load document image
        if CV2_AVAILABLE:
            doc_image = cv2.imread(document_image_path)
            doc_image = cv2.cvtColor(doc_image, cv2.COLOR_BGR2RGB)
        else:
            doc_image = np.array(Image.open(document_image_path).convert('RGB'))
        
        if doc_image is None:
            result["message"] = "Could not load document image"
            return result
        
        # Detect faces
        selfie_face_bbox = detect_face(selfie_image)
        doc_face_bbox = detect_face(doc_image)
        
        if selfie_face_bbox is None:
            result["message"] = "No face detected in selfie"
            result["details"]["selfie_face_detected"] = False
            return result
        
        if doc_face_bbox is None:
            result["message"] = "No face detected in document"
            result["details"]["document_face_detected"] = False
            return result
        
        result["details"]["selfie_face_detected"] = True
        result["details"]["document_face_detected"] = True
        
        # Extract face regions
        selfie_face = extract_face_region(selfie_image, selfie_face_bbox)
        doc_face = extract_face_region(doc_image, doc_face_bbox)
        
        # Compute embeddings
        selfie_embedding = compute_face_embedding(selfie_face)
        doc_embedding = compute_face_embedding(doc_face)
        
        # Compare faces
        score = compare_faces(selfie_embedding, doc_embedding)
        
        # Determine match (threshold: 60%)
        is_match = score >= 60.0
        
        # Convert to native Python types (for JSON serialization)
        result["success"] = True
        result["match"] = bool(is_match)  # Convert numpy bool to Python bool
        result["score"] = float(score)     # Convert numpy float to Python float
        
        if is_match:
            result["message"] = f"✅ Face match confirmed! Similarity: {score}%"
        else:
            result["message"] = f"❌ Face mismatch. Similarity: {score}% (threshold: 60%)"
        
        return result
        
    except Exception as e:
        logger.error(f"Face matching error: {e}")
        result["message"] = f"Error during face matching: {str(e)}"
        return result


# Demo function for frontend testing
def get_demo_match_result() -> Dict[str, Any]:
    """Return a demo match result for testing."""
    return {
        "success": True,
        "match": True,
        "score": 87.5,
        "message": "✅ Face match confirmed! Similarity: 87.5%",
        "details": {
            "selfie_face_detected": True,
            "document_face_detected": True,
            "algorithm": "Demo mode"
        }
    }
