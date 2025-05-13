import os
import requests
from dotenv import load_dotenv
import logging
from typing import Optional

# Load environment variables
load_dotenv()

MAILTRAP_API_TOKEN = os.getenv("MAILTRAP_API_TOKEN")
MAILTRAP_ENDPOINT = os.getenv("MAILTRAP_ENDPOINT")
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "no-reply@FilmOI.com")
SENDER_NAME = os.getenv("SENDER_NAME", "FilmOI")

# Setup logger
logger = logging.getLogger(__name__)

class EmailService:
    @staticmethod
    def send_verification_email(to_email: str, user_name: str, verification_link: str) -> bool:
        """
        Send email verification email using Mailtrap API
        """
        if not MAILTRAP_API_TOKEN or not MAILTRAP_ENDPOINT:
            logger.warning("Mailtrap credentials not configured. Email not sent.")
            return False
        
        headers = {
            "Authorization": f"Bearer {MAILTRAP_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Verify Your Email</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f9f9f9;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    padding: 20px 0;
                    border-bottom: 1px solid #eaeaea;
                }}
                .logo {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 10px;
                }}
                .logo svg {{
                    width: 32px;
                    height: 32px;
                    margin-right: 8px;
                    fill: #4f46e5; /* Primary color - adjust as needed */
                }}
                .logo-text {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4f46e5; /* Primary color - adjust as needed */
                }}
                .content {{
                    padding: 30px 20px;
                    text-align: center;
                }}
                h1 {{
                    color: #333;
                    font-size: 22px;
                    margin-bottom: 20px;
                }}
                p {{
                    color: #666;
                    line-height: 1.6;
                    margin-bottom: 20px;
                }}
                .button {{
                    display: inline-block;
                    background-color: #4f46e5; /* Primary color - adjust as needed */
                    color: white;
                    text-decoration: none;
                    padding: 12px 25px;
                    border-radius: 4px;
                    font-weight: bold;
                    margin: 20px 0;
                    transition: background-color 0.3s;
                }}
                .button:hover {{
                    background-color: #4338ca; /* Darker shade for hover */
                }}
                .footer {{
                    text-align: center;
                    padding-top: 20px;
                    border-top: 1px solid #eaeaea;
                    color: #999;
                    font-size: 12px;
                }}
                .help-text {{
                    font-size: 13px;
                    color: #888;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                            <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
                        </svg>
                        <span class="logo-text">FilmOI</span>
                    </div>
                </div>
                
                <div class="content">
                    <h1>Verifikasi Email Kamu</h1>
                    <p>Hai {user_name},</p>
                    <p>Terima kasih telah mendaftar di FilmOI! Untuk menyelesaikan pendaftaran, silakan verifikasi alamat email kamu dengan mengklik tombol di bawah ini.</p>
                    
                    <a href="{verification_link}" class="button">Verifikasi Email</a>
                    
                    <p class="help-text">Jika kamu tidak mendaftar di FilmOI, kamu bisa mengabaikan email ini.</p>
                    <p class="help-text">Link verifikasi ini akan kedaluwarsa dalam 24 jam.</p>
                </div>
                
                <div class="footer">
                    <p>© 2025 FilmOI. All rights reserved.</p>
                    <p>Jika kamu butuh bantuan, silakan balas email ini.</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        payload = {
            "from": {
                "email": SENDER_EMAIL,
                "name": SENDER_NAME
            },
            "to": [
                {
                    "email": to_email
                }
            ],
            "subject": "Verifikasi Email - FilmOI",
            "text": f"Hai {user_name}, terima kasih telah mendaftar di FilmOI! Silakan verifikasi email kamu dengan mengunjungi {verification_link}. Link ini akan kedaluwarsa dalam 24 jam.",
            "html": html_content,
            "category": "Verification"
        }
        
        try:
            response = requests.post(MAILTRAP_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Verification email sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send verification email: {str(e)}")
            return False

    @staticmethod
    def send_reset_password_email(to_email: str, user_name: str, reset_link: str) -> bool:
        """
        Send reset password email using Mailtrap API
        """
        if not MAILTRAP_API_TOKEN or not MAILTRAP_ENDPOINT:
            logger.warning("Mailtrap credentials not configured. Email not sent.")
            return False
        
        headers = {
            "Authorization": f"Bearer {MAILTRAP_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset Password</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f9f9f9;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    padding: 20px 0;
                    border-bottom: 1px solid #eaeaea;
                }}
                .logo {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 10px;
                }}
                .logo svg {{
                    width: 32px;
                    height: 32px;
                    margin-right: 8px;
                    fill: #4f46e5; /* Primary color - adjust as needed */
                }}
                .logo-text {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4f46e5; /* Primary color - adjust as needed */
                }}
                .content {{
                    padding: 30px 20px;
                    text-align: center;
                }}
                h1 {{
                    color: #333;
                    font-size: 22px;
                    margin-bottom: 20px;
                }}
                p {{
                    color: #666;
                    line-height: 1.6;
                    margin-bottom: 20px;
                }}
                .button {{
                    display: inline-block;
                    background-color: #4f46e5; /* Primary color - adjust as needed */
                    color: white;
                    text-decoration: none;
                    padding: 12px 25px;
                    border-radius: 4px;
                    font-weight: bold;
                    margin: 20px 0;
                    transition: background-color 0.3s;
                }}
                .button:hover {{
                    background-color: #4338ca; /* Darker shade for hover */
                }}
                .footer {{
                    text-align: center;
                    padding-top: 20px;
                    border-top: 1px solid #eaeaea;
                    color: #999;
                    font-size: 12px;
                }}
                .help-text {{
                    font-size: 13px;
                    color: #888;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                            <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
                        </svg>
                        <span class="logo-text">FilmOI</span>
                    </div>
                </div>
                
                <div class="content">
                    <h1>Reset Your Password</h1>
                    <p>Hai {user_name} yang manis,</p>
                    <p>Kami menerima permintaan untuk mengatur ulang password akun FilmOI kamu. Klik tombol di bawah untuk melanjutkan proses reset password.</p>
                    
                    <a href="{reset_link}" class="button">Reset Password</a>
                    
                    <p class="help-text">Jika kamu tidak meminta reset password, kamu bisa mengabaikan email ini dan tidak ada perubahan yang akan terjadi pada akun kamu.</p>
                    <p class="help-text">Link reset password ini akan kedaluwarsa dalam 24 jam.</p>
                </div>
                
                <div class="footer">
                    <p>© 2025 FilmOI. All rights reserved.</p>
                    <p>Jika kamu butuh bantuan, silakan balas email ini.</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        payload = {
            "from": {
                "email": SENDER_EMAIL,
                "name": SENDER_NAME
            },
            "to": [
                {
                    "email": to_email
                }
            ],
            "subject": "Password Reset Request",
            "text": f"Hello {user_name}, you have requested to reset your password. Please go to {reset_link} to reset it. This link will expire in 24 hours.",
            "html": html_content,
            "category": "Password Reset"
        }
        
        try:
            response = requests.post(MAILTRAP_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Password reset email sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    @staticmethod
    def send_welcome_email(to_email: str, user_name: str) -> bool:
        """
        Send welcome email to new user
        """
        if not MAILTRAP_API_TOKEN or not MAILTRAP_ENDPOINT:
            logger.warning("Mailtrap credentials not configured. Email not sent.")
            return False
        
        headers = {
            "Authorization": f"Bearer {MAILTRAP_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        html_content = f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Welcome to FilmOI</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f9f9f9;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #ffffff;
                    border-radius: 8px;
                    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                }}
                .header {{
                    text-align: center;
                    padding: 20px 0;
                    border-bottom: 1px solid #eaeaea;
                }}
                .logo {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-bottom: 10px;
                }}
                .logo svg {{
                    width: 32px;
                    height: 32px;
                    margin-right: 8px;
                    fill: #4f46e5;
                }}
                .logo-text {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4f46e5;
                }}
                .content {{
                    padding: 30px 20px;
                    text-align: center;
                }}
                h1 {{
                    color: #333;
                    font-size: 24px;
                    margin-bottom: 20px;
                }}
                p {{
                    color: #666;
                    line-height: 1.6;
                    margin-bottom: 20px;
                }}
                .button {{
                    display: inline-block;
                    background-color: #4f46e5;
                    color: white;
                    text-decoration: none;
                    padding: 12px 25px;
                    border-radius: 4px;
                    font-weight: bold;
                    margin: 20px 0;
                }}
                .footer {{
                    text-align: center;
                    padding-top: 20px;
                    border-top: 1px solid #eaeaea;
                    color: #999;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <div class="logo">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                            <path d="M2 5a2 2 0 012-2h7a2 2 0 012 2v4a2 2 0 01-2 2H9l-3 3v-3H4a2 2 0 01-2-2V5z" />
                            <path d="M15 7v2a4 4 0 01-4 4H9.828l-1.766 1.767c.28.149.599.233.938.233h2l3 3v-3h2a2 2 0 002-2V9a2 2 0 00-2-2h-1z" />
                        </svg>
                        <span class="logo-text">FilmOI</span>
                    </div>
                </div>
                
                <div class="content">
                    <h1>Welcome to FilmOI!</h1>
                    <p>Hai {user_name},</p>
                    <p>Terima kasih telah bergabung dengan FilmOI! Kami senang kamu ada di sini.</p>
                    <p>FilmOI adalah platform analisis risiko investasi film yang membantu kamu membuat keputusan investasi untuk proyek film dengan lebih cerdas.</p>
                    
                    <a href="https://FilmOI.com/dashboard" class="button">Mulai Menganalisis</a>
                    
                    <p>Jika kamu memiliki pertanyaan atau membutuhkan bantuan, jangan ragu untuk menghubungi tim dukungan kami.</p>
                </div>
                
                <div class="footer">
                    <p>© 2025 FilmOI. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        '''
        
        payload = {
            "from": {
                "email": SENDER_EMAIL,
                "name": SENDER_NAME
            },
            "to": [
                {
                    "email": to_email
                }
            ],
            "subject": "Welcome to FilmOI!",
            "text": f"Hello {user_name}, welcome to FilmOI! We're glad you're here.",
            "html": html_content,
            "category": "Welcome"
        }
        
        try:
            response = requests.post(MAILTRAP_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(f"Welcome email sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False