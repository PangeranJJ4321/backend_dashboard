import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException

load_dotenv()

# Setup konfigurasi Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_to_cloudinary(file: UploadFile):
    # Validasi tipe file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")

    try:
        result = cloudinary.uploader.upload(file.file)
        return result.get("secure_url")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload ke Cloudinary gagal: {str(e)}")

def delete_image_from_cloudinary(public_id: str):
    try:
        result = cloudinary.uploader.destroy(public_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal hapus gambar: {str(e)}")
