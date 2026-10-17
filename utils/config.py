from dotenv import load_dotenv
import os
import joblib
from supabase import create_client, Client, ClientOptions


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


# Seconds to wait on Supabase before giving up. Without this, a
# paused/sleeping free-tier project doesn't fail fast -- it hangs the
# request instead, which defeats the whole point of usage_tracker.py's
# try/except-and-fall-back-to-local-file design: that fallback only
# helps if the Supabase call actually raises in a reasonable time.
# This is what makes /predict/forest, /predict/xgboost, and /stats
# stall for a long time instead of falling back to the local file
# within a second or two.
_SUPABASE_TIMEOUT_SECONDS = 5

# Supabase client (used by usage_tracker.py). create_client() itself
# makes no network call, so constructing it eagerly is safe even if
# the project happens to be asleep right now -- only the .execute()
# calls in usage_tracker.py can time out, and that's what
# postgrest_client_timeout bounds.
supabase_client: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(
            SUPABASE_URL, SUPABASE_KEY,
            options=ClientOptions(postgrest_client_timeout=_SUPABASE_TIMEOUT_SECONDS),
        )
    except Exception as e:
        print(f"[config] Failed to initialize Supabase client: {e}")
else:
    print("[config] SUPABASE_URL / SUPABASE_KEY not set — usage tracking will fall back to local file only.")
