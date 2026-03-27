"""
config.py — Load all settings from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
VISION_MODEL    = os.getenv("VISION_MODEL", "gpt-4o")
IMAGE_MODEL     = os.getenv("IMAGE_MODEL", "dall-e-3")
IMAGE_SIZE      = os.getenv("IMAGE_SIZE", "1024x1024")
IMAGE_QUALITY   = os.getenv("IMAGE_QUALITY", "standard")
APP_TITLE       = os.getenv("APP_TITLE", "AI Vision Studio")
MAX_HISTORY     = int(os.getenv("MAX_HISTORY", "50"))
DATABASE_URL    = "sqlite:///./vision_studio.db"
  

