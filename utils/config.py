from dotenv import load_dotenv
import os
import joblib
from supabase import create_client, Client


load_dotenv(override=True)

# .env Variables
APP_NAME = os.getenv('APP_NAME')
VERSION = os.getenv('VERSION')
SECRET_KEY_TOKEN = os.getenv('SECRET_KEY_TOKEN')

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_FOLDER_PATH = os.path.join(BASE_DIR, "models")


# Models
preprocessor = joblib.load(os.path.join(MODELS_FOLDER_PATH, "preprocessor.pkl"))
forest_model = joblib.load(os.path.join(MODELS_FOLDER_PATH, "forest_tuned.pkl"))
xgboost_model = joblib.load(os.path.join(MODELS_FOLDER_PATH, "xgb-tuned.pkl"))


# Supabase client (used by usage_tracker.py)
supabase_client: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"[config] Failed to initialize Supabase client: {e}")
else:
    print("[config] SUPABASE_URL / SUPABASE_KEY not set — usage tracking will fall back to local file only.")