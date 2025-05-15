import os
import logging
import traceback
from typing import Dict, Any, Optional, Callable
from functools import wraps

# Set up a dedicated prediction logger
logger = logging.getLogger("prediction_debug")
logger.setLevel(logging.DEBUG)

# Configure logger if not already configured
if not logger.handlers:
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Add a file handler to log to a file
    file_handler = logging.FileHandler(os.path.join(log_dir, "prediction_debug.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Add console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

def debug_prediction_process(func: Callable) -> Callable:
    """
    Decorator to debug prediction functions and log detailed error information.
    
    Args:
        func: Function to wrap with debugging
        
    Returns:
        Wrapped function with debugging
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Filter out sensitive or large data from logs
            safe_args = ["<filtered>" if isinstance(arg, (dict, list)) and len(str(arg)) > 100 else arg for arg in args]
            safe_kwargs = {k: "<filtered>" if isinstance(v, (dict, list)) and len(str(v)) > 100 else v for k, v in kwargs.items()}
            
            logger.debug(f"Calling {func.__name__} with args: {safe_args}, kwargs: {safe_kwargs}")
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
    try:
        from app.utils.prediction import REGRESSION_MODEL_PATH, CLASSIFICATION_MODEL_PATH, BASE_DIR, MODEL_DIR
        
        logger.debug(f"Base directory: {BASE_DIR}")
        logger.debug(f"Models directory: {MODEL_DIR}")
        logger.debug(f"Regression model path: {REGRESSION_MODEL_PATH}")
        logger.debug(f"Classification model path: {CLASSIFICATION_MODEL_PATH}")
        
        # Check if directories exist
        logger.debug(f"Models directory exists: {os.path.exists(MODEL_DIR)}")
        
        regression_dir = os.path.join(MODEL_DIR, "regresi")
        classification_dir = os.path.join(MODEL_DIR, "klasifikasi")
        
        logger.debug(f"Regression directory exists: {os.path.exists(regression_dir)}")
        logger.debug(f"Classification directory exists: {os.path.exists(classification_dir)}")
        
        # Check if files exist
        logger.debug(f"Regression model exists: {os.path.exists(REGRESSION_MODEL_PATH)}")
        logger.debug(f"Classification model exists: {os.path.exists(CLASSIFICATION_MODEL_PATH)}")
        
        # Create directories if they don't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(regression_dir, exist_ok=True)
        os.makedirs(classification_dir, exist_ok=True)
        
        # List files in directories
        try:
            if os.path.exists(MODEL_DIR):
                logger.debug(f"Files in model directory: {os.listdir(MODEL_DIR)}")
            else:
                logger.debug("Model directory not found")
                
            if os.path.exists(regression_dir):
                logger.debug(f"Files in regression directory: {os.listdir(regression_dir)}")
            else:
                logger.debug("Regression directory not found")
                
            if os.path.exists(classification_dir):
                logger.debug(f"Files in classification directory: {os.listdir(classification_dir)}")
            else:
                logger.debug("Classification directory not found")
        except Exception as e:
            logger.error(f"Error listing directory contents: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in log_model_paths: {str(e)}")
        logger.error(traceback.format_exc())

def check_prediction_input(data: Dict[str, Any]) -> Optional[str]:
    """
    Validate prediction input data and return error message if invalid.
    
    Args:
        data: Prediction input data
    
    Returns:
        Error message if validation fails, None if valid
    """
    try:
        logger.debug(f"Validating prediction input: {list(data.keys())}")
        
        required_fields = ["film_title", "release_date", "budget", "genres"]
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logger.warning(error_msg)
            return error_msg
        
        # Check budget is positive
        if data.get("budget", 0) <= 0:
            error_msg = "Budget must be positive"
            logger.warning(error_msg)
            return error_msg
        
        # Check genres is a list
        if not isinstance(data.get("genres", []), list):
            error_msg = "Genres must be a list"
            logger.warning(error_msg)
            return error_msg
        
        logger.debug("Input validation successful")
        return None
        
    except Exception as e:
        logger.error(f"Error in check_prediction_input: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Validation error: {str(e)}"


def initialize_prediction_logging():
    """Initialize and configure all prediction-related loggers."""
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger for the prediction module
        root_logger = logging.getLogger("app.utils")
        root_logger.setLevel(logging.INFO)
        
        # Add file handler if not already added
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            file_handler = logging.FileHandler(os.path.join(log_dir, "prediction_system.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            root_logger.addHandler(file_handler)
        
        # Add console handler if not already added
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
            root_logger.addHandler(console_handler)
        
        # Configure specific loggers
        loggers = [
            "app.utils.prediction",
            "app.utils.prediction_debug",
            "app.services.prediction_service",
            "app.repositories.predictionRepositories",
            "app.controllers.predictionController"
        ]
        
        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            
            # Add handlers if needed
            if not logger.handlers:
                # File handler
                file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name.split('.')[-1]}.log"))
                file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(file_handler)
                
                # Console handler
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
                logger.addHandler(console_handler)
        
        logger.info("Prediction logging system initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing prediction logging: {str(e)}")
        traceback.print_exc()
        return False