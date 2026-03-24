import os
import cv2
import numpy as np
import urllib.request
import logging

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FAS_MODEL_PATH = os.path.join(MODEL_DIR, "antispoofing.onnx")
FAS_MODEL_URL = "https://github.com/RizwanMunawar/yolov8-face-antispoofing/raw/main/weights/antispoofing.onnx"

session = None

def load_fas_model():
    global session
    if not ONNX_AVAILABLE:
        return False
        
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(FAS_MODEL_PATH):
        logger.info(f"Downloading Face Anti-Spoofing ONNX model...")
        try:
            urllib.request.urlretrieve(FAS_MODEL_URL, FAS_MODEL_PATH)
            logger.info("ONNX Download complete.")
        except Exception as e:
            logger.error(f"Failed to download FAS model: {e}")
            return False
            
    if session is None:
        try:
            session = ort.InferenceSession(FAS_MODEL_PATH, providers=["CPUExecutionProvider"])
        except Exception as e:
            logger.error(f"Failed to load ONNX session: {e}")
            return False
    return True

def analyze_liveness(image_array: np.ndarray) -> dict:
    """
    Analyzes an image (numpy array BGR) for liveness/spoofing.
    Returns dict: {'is_real': bool, 'score': float, 'method': str, 'reason': str}
    """
    # 1. Base OpenCV Heuristics (Blur checking)
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if blur_score < 20.0:
        return {"is_real": False, "score": 0.0, "reason": "Image is too blurry or low quality", "method": "OpenCV"}
        
    # 2. Deep Learning ONNX Inference
    if load_fas_model():
        try:
            input_name = session.get_inputs()[0].name
            input_shape = session.get_inputs()[0].shape
            
            # Typically models expect 256x256 or similar
            h, w = 256, 256
            if len(input_shape) >= 4 and isinstance(input_shape[2], int):
                h, w = input_shape[2], input_shape[3]
                
            img_resized = cv2.resize(image_array, (w, h))
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Handle channels-first vs channels-last depending on model
            if input_shape[1] == 3:
                img_transposed = np.transpose(img_normalized, (2, 0, 1)) # HWC to CHW
            else:
                img_transposed = img_normalized
                
            img_expanded = np.expand_dims(img_transposed, axis=0) # Add batch dim
            
            outputs = session.run(None, {input_name: img_expanded})
            out = outputs[0][0]
            
            # Extract score
            if isinstance(out, (np.ndarray, list)) and len(out) >= 2:
                real_score = float(out[1]) # Usually 1 is real, 0 is spoof
            else:
                real_score = float(out) if not isinstance(out, (np.ndarray, list)) else float(out[0])
            
            # Softmax or probability threshold
            is_real = real_score > 0.7
            return {
                "is_real": is_real, 
                "score": real_score, 
                "reason": "Passed" if is_real else "Presentation Attack Detected (Confidence Required)",
                "method": "ONNX DeepLearning"
            }
        except Exception as e:
            logger.warning(f"ONNX inference shape mismatch or error: {e}. Falling back to OpenCV.")
            
    # 3. Fallback to Deep OpenCV Mathematical Heuristics (FFT Moiré Pattern Detection)
    # Screens emit highly unnatural high-frequency artificial grid patterns
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    rows, cols = gray.shape
    crow, ccol = rows//2, cols//2
    # Mask out the center (low frequencies/general shapes)
    magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30] = 0
    
    # Calculate energy in high frequencies
    high_freq_energy = np.sum(magnitude_spectrum) / (rows * cols)
    
    # Real faces have soft gradients (low high-freq energy)
    # Screens have pixel grids causing spikes in high-freq energy
    # Extremely bright or sharp highlights also cause this
    score = max(0.0, 1.0 - (high_freq_energy / 200.0))
    is_real = score > 0.6
    
    return {
        "is_real": is_real, 
        "score": float(score), 
        "reason": "Passed FFT analysis" if is_real else "Digital Screen Moiré Pattern Detected",
        "method": "OpenCV FFT Analysis"
    }
