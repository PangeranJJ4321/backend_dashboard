import os
import logging
import traceback
from typing import Dict, Any, Optional

# Set up a dedicated prediction logger
logger = logging.getLogger("prediction_debug")
logger.setLevel(logging.DEBUG)

# Add a file handler to log to a file
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "prediction_debug.log"))
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def debug_prediction_process(func):
    """
    Decorator to debug prediction functions and log detailed error information.
    
    Args:
        func: Function to wrap with debugging
        
    Returns:
        Wrapped function with debugging
    """
    def wrapper(*args, **kwargs):
        try:
            logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    return wrapper

def log_model_paths():
    """Log the model paths and file existence to debug loading issues."""
    from app.utils.prediction import REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, BASE_DIR, MODEL_DIR
    
    logger.debug(f"Base directory: {BASE_DIR}")
    logger.debug(f"Models directory: {MODEL_DIR}")
    logger.debug(f"Regression model path: {REGRESSION_MODEL_PATH}")
    logger.debug(f"Classification model path: {CLASSIFICATION_MODEL_PATH}")
    
    # Check if files exist
    logger.debug(f"Regression model exists: {os.path.exists(REGRESSION_MODEL_PATH)}")
    logger.debug(f"Classification model exists: {os.path.exists(CLASSIFICATION_MODEL_PATH)}")
    
    # List files in directories
    logger.debug(f"Files in model directory: {os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else 'Directory not found'}")
    regression_dir = os.path.join(MODEL_DIR, "regresi")
    classification_dir = os.path.join(MODEL_DIR, "klasifikasi")
    
    logger.debug(f"Files in regression directory: {os.listdir(regression_dir) if os.path.exists(regression_dir) else 'Directory not found'}")
    logger.debug(f"Files in classification directory: {os.listdir(classification_dir) if os.path.exists(classification_dir) else 'Directory not found'}")

def check_prediction_input(data: Dict[str, Any]) -> Optional[str]:
    """
    Validate prediction input data and return error message if invalid.
    
    Args:
        data: Prediction input data
    
    Returns:
        Error message if validation fails, None if valid
    """
    required_fields = ["film_title", "release_date", "budget", "genres"]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return f"Missing required fields: {', '.join(missing_fields)}"
    
    # Check budget is positive
    if data.get("budget", 0) <= 0:
        return "Budget must be positive"
    
    # Check genres is a list
    if not isinstance(data.get("genres", []), list):
        return "Genres must be a list"
    
    return None