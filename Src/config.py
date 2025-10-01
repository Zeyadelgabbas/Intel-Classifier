import os 
from dotenv import load_dotenv
import tensorflow as tf 
import joblib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN messages
tf.get_logger().setLevel('ERROR')
load_dotenv(override=True)
APP_NAME = os.getenv('APP_NAME')
API_SECRET_KEY = os.getenv('API_SECRET_KEY')
VERSION = os.getenv('VERSION')

SRC_FOLDER_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL =tf.keras.models.load_model(os.path.join(os.getcwd(),'artifacts','model.keras'))
IDX2LABEL = joblib.load(os.path.join(os.getcwd(),'artifacts','idx2label.joblib'))

DOWNLOADED_IMAGES_PATHS = os.path.join(os.getcwd(),'artifacts','inference_images')
os.makedirs(DOWNLOADED_IMAGES_PATHS, exist_ok=True)
TARGET_SIZE = (100,100)

def ensure_directories():
    os.makedirs(DOWNLOADED_IMAGES_PATHS, exist_ok=True)