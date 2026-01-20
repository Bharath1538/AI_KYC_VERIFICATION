"""
Real-Time Aadhaar Card Verification System

Live video pipeline that:
1. Shows bounding box overlay for card placement
2. Automatically detects Aadhaar cards using Id_Classifier
3. Assesses image quality (blur, brightness, contrast, glare)
4. Auto-captures when detection is stable and quality is acceptable
5. Runs OCR extraction using existing inference.py
6. Verifies against backend database (placeholder for now)

Usage:
    python aadhaar_live_verification.py

Controls:
    q - Quit the application
    r - Reset/restart verification process
"""

import cv2
import numpy as np
import tempfile
import os
import time
import logging
import re
import argparse
from datetime import datetime
from enum import Enum
from typing import Tuple, Dict, List, Optional, Any
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import existing inference module (for Aadhaar detection only)
try:
    from inference import process_id
    INFERENCE_AVAILABLE = True
except ImportError:
    logger.warning("inference.py not found. Running in demo mode without actual Aadhaar detection.")
    INFERENCE_AVAILABLE = False

# Import Surya OCR modules (for text extraction)
try:
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    SURYA_AVAILABLE = True
    logger.info("Surya OCR loaded successfully.")
except ImportError as e:
    logger.warning(f"Surya OCR not available: {e}. OCR will run in demo mode.")
    SURYA_AVAILABLE = False


class VerificationState(Enum):
    """States for the verification state machine"""
    WAITING = "waiting"           # Waiting for card to be placed
    DETECTING = "detecting"       # Card detected, checking quality
    STABILIZING = "stabilizing"   # Good detection, waiting for stability
    CAPTURING = "capturing"       # Capturing the image
    VERIFYING = "verifying"       # Running OCR and backend verification
    SUCCESS = "success"           # Verification successful
    FAILED = "failed"             # Verification failed


class ImageQualityAssessor:
    """Handles image quality assessment for OCR suitability"""
    
    def __init__(
        self,
        min_sharpness: float = 100.0,
        min_brightness: float = 50.0,
        max_brightness: float = 220.0,
        min_contrast: float = 30.0,
        max_glare_ratio: float = 0.05,
        min_width: int = 300,
        min_height: int = 180
    ):
        self.min_sharpness = min_sharpness
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.max_glare_ratio = max_glare_ratio
        self.min_width = min_width
        self.min_height = min_height
    
    def assess(self, image: np.ndarray) -> Tuple[bool, float, List[str], Dict[str, Any]]:
        """
        Assess image quality for OCR suitability.
        
        Returns:
            Tuple of (is_acceptable, quality_score, issues_list, detailed_scores)
        """
        issues = []
        scores = {}
        
        if image is None or image.size == 0:
            return False, 0.0, ["Invalid image"], {}
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        
        # 1. BLUR DETECTION (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores['sharpness'] = laplacian_var
        if laplacian_var < self.min_sharpness:
            issues.append(f"Image too blurry (sharpness: {laplacian_var:.1f}, need: {self.min_sharpness})")
        
        # 2. BRIGHTNESS CHECK
        brightness = np.mean(gray)
        scores['brightness'] = brightness
        if brightness < self.min_brightness:
            issues.append(f"Image too dark (brightness: {brightness:.1f})")
        elif brightness > self.max_brightness:
            issues.append(f"Image overexposed (brightness: {brightness:.1f})")
        
        # 3. CONTRAST CHECK
        contrast = gray.std()
        scores['contrast'] = contrast
        if contrast < self.min_contrast:
            issues.append(f"Low contrast (contrast: {contrast:.1f})")
        
        # 4. RESOLUTION CHECK
        scores['resolution'] = (w, h)
        if w < self.min_width or h < self.min_height:
            issues.append(f"Resolution too low ({w}x{h}), need at least {self.min_width}x{self.min_height}")
        
        # 5. GLARE/OVEREXPOSURE DETECTION
        _, bright_regions = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
        glare_ratio = np.sum(bright_regions > 0) / (w * h)
        scores['glare_ratio'] = glare_ratio
        if glare_ratio > self.max_glare_ratio:
            issues.append(f"Glare detected ({glare_ratio*100:.1f}% of image)")
        
        # Calculate overall quality score (0-100)
        sharpness_score = min(100, (laplacian_var / 5))  # Max at 500
        brightness_score = 100 - abs(128 - brightness) * 0.8  # Best at 128
        contrast_score = min(100, contrast * 1.25)  # Max at 80
        glare_score = (1 - glare_ratio) * 100
        
        quality_score = (
            sharpness_score * 0.40 +   # Sharpness is most important
            brightness_score * 0.20 +
            contrast_score * 0.25 +
            glare_score * 0.15
        )
        quality_score = max(0, min(100, quality_score))
        scores['overall'] = quality_score
        
        is_acceptable = len(issues) == 0 and quality_score >= 60
        
        return is_acceptable, quality_score, issues, scores


class AadhaarDetector:
    """Handles Aadhaar card detection using the Id_Classifier model"""
    
    def __init__(self, min_confidence: float = 0.85):
        self.min_confidence = min_confidence
        self._temp_files = []
    
    def detect(self, image: np.ndarray) -> Tuple[bool, str, float]:
        """
        Detect if the image contains an Aadhaar card.
        
        Returns:
            Tuple of (is_aadhaar, doc_type, confidence)
        """
        if not INFERENCE_AVAILABLE:
            # Demo mode - simulate detection
            return True, "Aadhaar", 0.95
        
        # Save image to temp file for the classifier
        temp_path = tempfile.mktemp(suffix='.jpg')
        self._temp_files.append(temp_path)
        cv2.imwrite(temp_path, image)
        
        try:
            result = process_id(temp_path, classify_only=True)
            doc_type = result.get("doc_type", "Unknown")
            confidence = result.get("confidence", 0.0)
            
            is_aadhaar = (doc_type == "Aadhaar" and confidence >= self.min_confidence)
            
            return is_aadhaar, doc_type, confidence
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return False, "Error", 0.0
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    self._temp_files.remove(temp_path)
                except:
                    pass
    
    def cleanup(self):
        """Clean up any remaining temp files"""
        for f in self._temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
        self._temp_files = []


class OCRExtractor:
    """
    Handles OCR extraction from Aadhaar card images using Surya OCR.
    
    Surya OCR Modules Used:
    -----------------------
    - surya.foundation.FoundationPredictor: 
        Base predictor that initializes the shared model backbone and handles
        common configuration for all Surya tasks.
    
    - surya.recognition.RecognitionPredictor:
        Performs text recognition on detected text regions. Uses the foundation
        model to recognize characters and words in the detected bounding boxes.
    
    - surya.detection.DetectionPredictor:
        Detects text bounding boxes in images. Identifies regions containing text
        that should be passed to the recognition model.
    
    Workflow:
    1. DetectionPredictor finds text regions in the image
    2. RecognitionPredictor extracts text from each detected region
    3. Results are combined and parsed using regex patterns
    """
    
    def __init__(self):
        """Initialize Surya OCR models (loaded once for efficiency)."""
        if SURYA_AVAILABLE:
            try:
                logger.info("Initializing Surya OCR models...")
                self.foundation_predictor = FoundationPredictor()
                self.recognition_predictor = RecognitionPredictor(self.foundation_predictor)
                self.detection_predictor = DetectionPredictor()
                logger.info("Surya OCR models initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Surya OCR: {e}")
                self.foundation_predictor = None
                self.recognition_predictor = None
                self.detection_predictor = None
        else:
            self.foundation_predictor = None
            self.recognition_predictor = None
            self.detection_predictor = None
    
    def extract(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text from Aadhaar card image using Surya OCR.
        
        Args:
            image_path: Path to the Aadhaar card image file.
        
        Returns:
            Dictionary with extracted fields. Contains 'error' key if validation fails.
        """
        if not SURYA_AVAILABLE or self.recognition_predictor is None:
            # Demo mode - return sample data
            logger.info("Running OCR in demo mode (Surya not available)")
            return {
                "Name": "Demo User",
                "DOB": "01/01/1990",
                "Aadhaar Number": "XXXX XXXX XXXX",
                "Gender": "Male"
            }
        
        try:
            # Load image using PIL
            image = Image.open(image_path).convert("RGB")
            
            # Run Surya OCR: detection + recognition
            predictions = self.recognition_predictor(
                [image], 
                det_predictor=self.detection_predictor
            )
            
            # Extract all text lines from OCR result
            ocr_text = " ".join([line.text for line in predictions[0].text_lines])
            logger.info(f"OCR extracted text length: {len(ocr_text)}")
            
            # Validate if it is the Front Side of an Aadhaar card
            is_front, reason = self.is_front_side(ocr_text)
            if not is_front:
                logger.warning(f"Document validation failed: {reason}")
                return {"error": f"Validation Failed: {reason}", "raw_text": ocr_text}
            
            # Parse Aadhaar fields using regex
            result = self._parse_aadhaar_fields(ocr_text)
            result["validation"] = "Passed"
            return result
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return {"error": str(e)}

    def is_front_side(self, text: str) -> Tuple[bool, str]:
        """
        Determines if the text belongs to the FRONT side of an Aadhaar card.
        
        Returns:
            Tuple (is_front, reason)
        """
        text_lower = text.lower()
        
        # Strong indicators for BACK side
        # "Address", "Unique Identification Authority" (often on header back), "1947"
        back_keywords = ['address', 'address:', 'p.o. box', 's/o', 'w/o', 'd/o', '1947', 'help@uidai', 'www.uidai']
        for k in back_keywords:
            if k in text_lower:
                return False, f"Detected Back Side content ('{k}')"
            
        # Indicators for FRONT side
        # "Government of India", "DOB"/"Year of Birth", "Male"/"Female"
        front_keywords = ['government of india', 'govt of india', 'भारत सरकार', 'dob', 'date of birth', 'year of birth', 'male', 'female', 'पुरुष', 'महिला']
        
        # Count matches
        matches = sum(1 for k in front_keywords if k in text_lower)
        
        # We need at least 2 strong indicators to be confident it's the front
        if matches >= 2:
            return True, "Front Side Verified"
            
        return False, "Not a clear Aadhaar Front Side (missing DOB/Gender/Header)"
    
    def _parse_aadhaar_fields(self, ocr_text: str) -> Dict[str, Any]:
        """
        Parse Aadhaar card fields from OCR text using regex patterns.
        
        Patterns used:
        - Aadhaar Number: 12 digits in XXXX XXXX XXXX format
        - DOB: DD/MM/YYYY or DD-MM-YYYY format
        - Gender: MALE/FEMALE or Hindi equivalents (पुरुष/महिला)
        - Name: Text following specific keywords
        """
        result = {
            "Name": None,
            "DOB": None,
            "Aadhaar Number": None,
            "Gender": None,
            "raw_text": ocr_text
        }
        
        # Extract Aadhaar number (12 digits, may have spaces)
        aadhaar_match = re.search(r"\b\d{4}\s?\d{4}\s?\d{4}\b", ocr_text)
        if aadhaar_match:
            result["Aadhaar Number"] = aadhaar_match.group(0)
        
        # Extract DOB (DD/MM/YYYY or DD-MM-YYYY)
        dob_match = re.search(r"(\d{2}[/-]\d{2}[/-]\d{4})", ocr_text)
        if dob_match:
            result["DOB"] = dob_match.group(1)
        
        # Extract Gender (English and Hindi)
        gender_match = re.search(r"\b(MALE|FEMALE|पुरुष|महिला)\b", ocr_text, re.IGNORECASE)
        if gender_match:
            result["Gender"] = self._normalize_gender(gender_match.group(1))
        
        # Extract Name (heuristic approach)
        result["Name"] = self._extract_name(ocr_text)
        
        return result
    
    def _normalize_gender(self, gender: str) -> str:
        """Normalize gender to English (Male/Female)."""
        gender_lower = gender.lower()
        if gender_lower in ["male", "पुरुष"]:
            return "Male"
        elif gender_lower in ["female", "महिला"]:
            return "Female"
        return gender.capitalize()
    
    def _extract_name(self, ocr_text: str) -> Optional[str]:
        """
        Extract name from OCR text using heuristics.
        
        Looks for text patterns commonly found on Aadhaar cards
        such as text after the English name transliteration.
        """
        # Try to find name after common patterns
        # Pattern 1: Look for lines that might be names (capitalized words)
        lines = ocr_text.split()
        
        # Filter out known non-name text
        skip_words = {
            'government', 'india', 'aadhaar', 'dob', 'male', 'female',
            'भारत', 'सरकार', 'आधार', 'मेरा', 'मेरी', 'पहचान',
            'जन्म', 'तिथि', 'पुरुष', 'महिला'
        }
        
        # Look for potential name candidates (proper case words)
        name_candidates = []
        for word in lines:
            # Skip numbers and known keywords
            if word.lower() in skip_words:
                continue
            if re.match(r'^\d', word):  # Skip if starts with number
                continue
            # Check if it looks like a name (title case, reasonable length)
            if len(word) > 2 and word[0].isupper():
                name_candidates.append(word)
        
        # Return first few candidates as potential name
        if name_candidates:
            # Take up to 3 words that might form a name
            potential_name = ' '.join(name_candidates[:3])
            # Filter out if it looks like "Government of India" pattern
            if 'Government' not in potential_name and 'India' not in potential_name:
                return potential_name
        
        return None


class BackendVerifier:
    """
    Placeholder for backend database verification.
    Replace this with your actual PostgreSQL integration.
    """
    
    def __init__(self, db_config: Optional[Dict] = None):
        """
        Initialize backend connection.
        
        Args:
            db_config: Database configuration dictionary with keys:
                       host, port, database, user, password
        """
        self.db_config = db_config or {}
        self.connected = False
        
        # TODO: Replace with actual PostgreSQL connection
        # Example:
        # import psycopg2
        # self.conn = psycopg2.connect(**db_config)
        
        logger.info("Backend verifier initialized (placeholder mode)")
    
    def verify(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify extracted Aadhaar data against the backend database.
        
        Args:
            extracted_data: Dictionary with extracted fields from OCR
        
        Returns:
            Dictionary with verification result:
            {
                "verified": bool,
                "user_data": dict or None,
                "message": str
            }
        """
        # TODO: Replace with actual database query
        # Example:
        # cursor = self.conn.cursor()
        # cursor.execute(
        #     "SELECT * FROM users WHERE aadhaar_number = %s",
        #     (extracted_data.get("Aadhaar Number", "").replace(" ", ""),)
        # )
        # result = cursor.fetchone()
        
        # Placeholder implementation
        aadhaar_number = extracted_data.get("Aadhaar Number", "")
        
        if aadhaar_number and aadhaar_number != "XXXX XXXX XXXX":
            return {
                "verified": True,
                "user_data": {
                    "aadhaar": aadhaar_number,
                    "name": extracted_data.get("Name", "Unknown"),
                    "dob": extracted_data.get("DOB", "Unknown"),
                    "registered": True
                },
                "message": "User verified successfully (placeholder)"
            }
        else:
            return {
                "verified": False,
                "user_data": None,
                "message": "Aadhaar number not found in database (placeholder)"
            }
    
    def close(self):
        """Close database connection"""
        # TODO: Close actual connection
        # self.conn.close()
        pass


class LiveAadhaarVerifier:
    """
    Main class for live Aadhaar card verification.
    Handles video capture, detection, quality assessment, and verification pipeline.
    """
    
    def __init__(
        self,
        camera_id: Any = 1,
        width: int = 1280,
        height: int = 720,
        min_aadhaar_confidence: float = 0.85,
        min_quality_score: float = 60.0,
        stability_frames: int = 5,
        detection_interval_ms: int = 200,
        db_config: Optional[Dict] = None
    ):
        """
        Initialize the live verifier.
        
        Args:
            camera_id: Camera device ID (int) or URL (str)
            width: Desired camera width
            height: Desired camera height
            min_aadhaar_confidence: Minimum confidence for Aadhaar classification
            min_quality_score: Minimum image quality score (0-100)
            stability_frames: Number of consecutive good frames before capture
            detection_interval_ms: Milliseconds between detection runs
            db_config: Database configuration for backend verification
        """
        self.camera_id = camera_id
        self.min_quality_score = min_quality_score
        self.stability_frames = stability_frames
        self.detection_interval_ms = detection_interval_ms
        
        # Initialize components
        self.quality_assessor = ImageQualityAssessor()
        self.detector = AadhaarDetector(min_confidence=min_aadhaar_confidence)
        self.ocr_extractor = OCRExtractor()
        self.backend_verifier = BackendVerifier(db_config)
        
        # State management
        self.state = VerificationState.WAITING
        self.stable_count = 0
        self.last_detection_time = 0
        self.current_issues = []
        self.last_result = None
        
        # Video capture
        self.cap = None
        self.frame_width = width
        self.frame_height = height
        
        # Bounding box dimensions (Aadhaar aspect ratio: 85.6mm x 53.98mm ≈ 1.586:1)
        self.box_width = 400
        self.box_height = int(self.box_width / 1.586)
        self.box_x = 0
        self.box_y = 0
        
        # Colors
        self.COLOR_WAITING = (200, 200, 200)   # Light gray
        self.COLOR_DETECTING = (0, 255, 255)   # Yellow
        self.COLOR_GOOD = (0, 255, 0)          # Green
        self.COLOR_ERROR = (0, 0, 255)         # Red
        self.COLOR_SUCCESS = (0, 255, 0)       # Green
    
    def _init_camera(self) -> bool:
        """Initialize video capture"""
        import platform
        # Use platform-specific backend: CAP_DSHOW for Windows, default for Mac/Linux
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera {self.camera_id}")
            return False
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        # Get actual resolution
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate bounding box position (centered)
        self.box_x = (self.frame_width - self.box_width) // 2
        self.box_y = (self.frame_height - self.box_height) // 2
        
        logger.info(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        return True
    
    def _get_roi(self, frame: np.ndarray) -> np.ndarray:
        """Extract the region of interest (bounding box area)"""
        return frame[self.box_y:self.box_y + self.box_height, 
                     self.box_x:self.box_x + self.box_width].copy()
    
    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding box and status overlay on the frame"""
        overlay = frame.copy()
        
        # Determine color based on state
        color = self.COLOR_WAITING
        if self.state == VerificationState.DETECTING:
            color = self.COLOR_DETECTING
        elif self.state == VerificationState.STABILIZING:
            color = self.COLOR_GOOD
        elif self.state in [VerificationState.SUCCESS]:
            color = self.COLOR_SUCCESS
        elif self.state == VerificationState.FAILED:
            color = self.COLOR_ERROR
        
        # Draw main bounding box
        cv2.rectangle(overlay, 
                      (self.box_x, self.box_y), 
                      (self.box_x + self.box_width, self.box_y + self.box_height), 
                      color, 3)
        
        # Draw corner markers
        corner_len = 25
        thickness = 5
        corners = [
            (self.box_x, self.box_y),                                    # Top-left
            (self.box_x + self.box_width, self.box_y),                   # Top-right
            (self.box_x, self.box_y + self.box_height),                  # Bottom-left
            (self.box_x + self.box_width, self.box_y + self.box_height)  # Bottom-right
        ]
        
        for i, (cx, cy) in enumerate(corners):
            # Horizontal line
            dx = corner_len if i % 2 == 0 else -corner_len
            cv2.line(overlay, (cx, cy), (cx + dx, cy), color, thickness)
            # Vertical line
            dy = corner_len if i < 2 else -corner_len
            cv2.line(overlay, (cx, cy), (cx, cy + dy), color, thickness)
        
        # Status messages
        status_messages = {
            VerificationState.WAITING: "Place your Aadhaar card inside the box",
            VerificationState.DETECTING: "Verifying document...",
            VerificationState.STABILIZING: f"Hold steady... ({self.stable_count}/{self.stability_frames})",
            VerificationState.CAPTURING: "Capturing image...",
            VerificationState.VERIFYING: "Processing... Please wait",
            VerificationState.SUCCESS: "Verification successful!",
            VerificationState.FAILED: "Verification failed. Please try again."
        }
        
        main_status = status_messages.get(self.state, "")
        
        # Draw semi-transparent status bar at top
        cv2.rectangle(overlay, (0, 0), (self.frame_width, 80), (0, 0, 0), -1)
        overlay = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw status text
        cv2.putText(overlay, main_status, (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Draw issues/errors
        if self.current_issues:
            issue_text = " | ".join(self.current_issues[:2])  # Show max 2 issues
            cv2.putText(overlay, issue_text, (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        
        # Draw keyboard shortcuts at bottom
        cv2.rectangle(overlay, (0, self.frame_height - 30), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.putText(overlay, "Press 'q' to quit | 'r' to reset", 
                    (20, self.frame_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Show last result if available
        if self.last_result and self.state == VerificationState.SUCCESS:
            self._draw_result_overlay(overlay)
        
        return overlay
    
    def _draw_result_overlay(self, frame: np.ndarray):
        """Draw the extraction result on the frame"""
        if not self.last_result:
            return
        
        # Draw result panel on the right side
        panel_x = self.frame_width - 350
        panel_y = 100
        panel_width = 330
        panel_height = 200
        
        # Semi-transparent background
        cv2.rectangle(frame, (panel_x, panel_y), 
                      (panel_x + panel_width, panel_y + panel_height), 
                      (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "Extracted Data:", (panel_x + 10, panel_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Fields
        y_offset = 55
        for key, value in list(self.last_result.items())[:5]:
            text = f"{key}: {value}"
            if len(text) > 35:
                text = text[:32] + "..."
            cv2.putText(frame, text, (panel_x + 10, panel_y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
    
    def _should_run_detection(self) -> bool:
        """Check if enough time has passed to run detection"""
        current_time = time.time() * 1000  # Convert to ms
        if current_time - self.last_detection_time >= self.detection_interval_ms:
            self.last_detection_time = current_time
            return True
        return False
    
    def _process_frame(self, frame: np.ndarray) -> None:
        """Process a single frame through the verification pipeline"""
        if not self._should_run_detection():
            return
        
        roi = self._get_roi(frame)
        
        # Check if ROI has enough content (not empty/uniform)
        if roi.std() < 10:  # Very low variance = likely no card
            self.state = VerificationState.WAITING
            self.stable_count = 0
            self.current_issues = ["No card detected in the box"]
            return
        
        # Step 1: Check if it's an Aadhaar card
        is_aadhaar, doc_type, confidence = self.detector.detect(roi)
        
        if not is_aadhaar:
            self.state = VerificationState.DETECTING
            self.stable_count = 0
            if doc_type == "Error":
                self.current_issues = ["Detection error occurred"]
            elif confidence > 0:
                self.current_issues = [f"Detected: {doc_type} (need Aadhaar card)"]
            else:
                self.current_issues = ["Not a valid Aadhaar card"]
            return
        
        # Step 2: Check image quality
        is_quality_ok, quality_score, issues, _ = self.quality_assessor.assess(roi)
        
        if not is_quality_ok:
            self.state = VerificationState.DETECTING
            self.stable_count = 0
            self.current_issues = issues[:2]  # Show top 2 issues
            return
        
        # Step 3: Good frame detected - increment stability counter
        self.current_issues = [f"Quality: {quality_score:.0f}% | Conf: {confidence:.0%}"]
        self.state = VerificationState.STABILIZING
        self.stable_count += 1
        
        # Step 4: If stable enough, capture and verify
        if self.stable_count >= self.stability_frames:
            self._capture_and_verify(roi)
    
    def _capture_and_verify(self, roi: np.ndarray) -> None:
        """Capture the ROI and run full verification pipeline"""
        self.state = VerificationState.CAPTURING
        
        # Save captured image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        capture_dir = os.path.join(os.path.dirname(__file__), "captures")
        os.makedirs(capture_dir, exist_ok=True)
        capture_path = os.path.join(capture_dir, f"aadhaar_{timestamp}.jpg")
        cv2.imwrite(capture_path, roi)
        logger.info(f"Captured image saved to: {capture_path}")
        
        self.state = VerificationState.VERIFYING
        
        # Run OCR extraction
        extracted_data = self.ocr_extractor.extract(capture_path)
        logger.info(f"Extracted data: {extracted_data}")
        
        if "error" in extracted_data:
            self.state = VerificationState.FAILED
            self.current_issues = [f"OCR Error: {extracted_data['error']}"]
            self.stable_count = 0
            return
        
        # Run backend verification
        verification_result = self.backend_verifier.verify(extracted_data)
        logger.info(f"Verification result: {verification_result}")
        
        if verification_result["verified"]:
            self.state = VerificationState.SUCCESS
            self.last_result = extracted_data
            self.current_issues = [verification_result["message"]]
        else:
            self.state = VerificationState.FAILED
            self.current_issues = [verification_result["message"]]
            self.stable_count = 0
    
    def reset(self):
        """Reset the verification state"""
        self.state = VerificationState.WAITING
        self.stable_count = 0
        self.current_issues = []
        self.last_result = None
        logger.info("Verification reset")
    
    def run(self):
        """Main loop - run the live verification"""
        if not self._init_camera():
            logger.error("Failed to initialize camera. Exiting.")
            return
        
        logger.info("Starting live Aadhaar verification. Press 'q' to quit, 'r' to reset.")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                
                # Flip horizontally for mirror effect (more intuitive for users)
                frame = cv2.flip(frame, 1)
                
                # Process frame if not in terminal state
                if self.state not in [VerificationState.SUCCESS, VerificationState.CAPTURING, 
                                        VerificationState.VERIFYING]:
                    self._process_frame(frame)
                
                # Draw overlay
                display_frame = self._draw_overlay(frame)
                
                # Show frame
                cv2.imshow("Aadhaar Live Verification", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested by user")
                    break
                elif key == ord('r'):
                    self.reset()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.detector.cleanup()
        self.backend_verifier.close()
        logger.info("Cleanup complete")


def main():
    """Entry point for the verification system"""
    parser = argparse.ArgumentParser(description="Real-Time Aadhaar Verification System")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0). Try 1 or 2 for DroidCam.")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    
    args = parser.parse_args()

    print("=" * 60)
    print("  REAL-TIME AADHAAR CARD VERIFICATION SYSTEM")
    print("=" * 60)
    print()
    print("Instructions:")
    print("  1. Place your Aadhaar card inside the bounding box")
    print("  2. Hold steady until the system captures the image")
    print("  3. Wait for verification to complete")
    print()
    print("Controls:")
    print("  q - Quit the application")
    print("  r - Reset/restart verification")
    print()
    print("=" * 60)
    print(f"Camera ID: {args.camera}")
    print()
    
    # Configuration - adjust these as needed
    config = {
        "camera_id": args.camera,
        "width": args.width,
        "height": args.height,
        "min_aadhaar_confidence": 0.85,
        "min_quality_score": 60.0,
        "stability_frames": 5,
        "detection_interval_ms": 300,  # Run detection every 300ms to reduce load
        "db_config": {
            # TODO: Add your PostgreSQL connection details
            # "host": "localhost",
            # "port": 5432,
            # "database": "aadhaar_db",
            # "user": "your_user",
            # "password": "your_password"
            
        }
    }
    
    verifier = LiveAadhaarVerifier(**config)
    verifier.run()


if __name__ == "__main__":
    main()
