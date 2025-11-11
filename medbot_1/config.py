import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash-lite"

# OpenAI API Configuration (for Whisper STT, etc.)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google Cloud Configuration (for TTS)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

# Ngrok Configuration
# You should store your actual ngrok URL in an environment variable called NGROK_URL
NGROK_URL = os.getenv("NGROK_URL", "https://561ce706eda5.ngrok-free.app")

# Voice Settings
VOICE_SAMPLE_RATE = 16000
VOICE_CHANNELS = 1
