import os
import requests
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Get Mailtrap credentials
MAILTRAP_API_TOKEN = os.getenv("MAILTRAP_API_TOKEN")
MAILTRAP_ENDPOINT = os.getenv("MAILTRAP_ENDPOINT")

# Setup logger
logger = logging.getLogger(__name__)

def send_test_email(to_email: str) -> bool:
    """
    Send test email to verify Mailtrap configuration
    """
    if not MAILTRAP_API_TOKEN or not MAILTRAP_ENDPOINT:
        logger.error("Mailtrap credentials not configured")
        return False
    
    headers = {
        "Authorization": f"Bearer {MAILTRAP_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "from": {
            "email": "hello@chatoi.com",
            "name": "ChatOI Test"
        },
        "to": [
            {
                "email": to_email
            }
        ],
        "subject": "ChatOI API Test Email",
        "text": "Congratulations! If you're reading this, your Mailtrap integration is working correctly.",
        "category": "Integration Test"
    }
    
    try:
        response = requests.post(MAILTRAP_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        logger.info(f"Test email sent to {to_email}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send test email: {str(e)}")
        return False

# Add this code to test the Mailtrap integration
if __name__ == "__main__":
    recipient = input("Enter recipient email to test Mailtrap: ")
    result = send_test_email(recipient)
    print(f"Email sent: {result}")