from fastapi import FastAPI , Depends , HTTPException,UploadFile,File
from fastapi.security import APIKeyHeader
import uvicorn
import os 
from Src.inference import ImageClassifier
from Src.logger import get_logger
from Src.config import API_SECRET_KEY , APP_NAME , VERSION , DOWNLOADED_IMAGES_PATHS 
from Src.schemas import PredictionsResponse
from typing import List
from Src.utils import ensure_directories , delete_files
logging = get_logger(__name__)


app = FastAPI(title=APP_NAME,version=VERSION)
api_key_header=APIKeyHeader(name='X-API-Key')
classifier = ImageClassifier()  # Holds the model 

async def check_api_key(api_key:str = Depends(api_key_header)):
    if api_key != API_SECRET_KEY:
        raise HTTPException(status_code=403 , detail="Not authorized")
    else:
        return api_key
    
@app.get('/' ,tags=['Home'])
def home(api_key:str = Depends(check_api_key)):
    return {'message':f'Welcome to {APP_NAME}'}

@app.post('/Classify-batches-memory',tags=['Model'],response_model=PredictionsResponse)
async def classify_batch_memory(files: List[UploadFile] = File(...) , api_key:str = Depends(check_api_key)):
    """Classify multiple images in a batch and images are stored in memory (for small sizes batches)"""
    try:
        if not files:
            raise HTTPException(status_code=400,detail="No Uploaded files!")
        else:
            content = []
            for file in files:
                logging.info("entered file for files")
                if not file.content_type or not file.content_type.startswith('image/'):
                    logging.info("entered type check of file")
                    raise HTTPException(status_code=400,detail="uploaded file isn't image")
                
                file_bytes = await file.read()
                content.append(file_bytes)
            predictions = classifier.predict_batch(images=content)
            return predictions
    except HTTPException:  # let FastAPI handle this one
        raise
    except Exception as e:
        logging.error(f"error in memory prediction endpoint {str(e)} ")
        raise HTTPException(500,detail="Error making predictions")
    
@app.post('/Classify-batches-disk',tags = ['Model'],response_model=PredictionsResponse)
async def classify_batch_disk(files:List[UploadFile] = File(...),api_key :str= Depends(check_api_key)):
    """Classify multiple images in a batch and images are saved on disk (for big sizes batches)"""
    if not files:
        raise HTTPException(status_code=400,detail = "No uploaded files")
    else:
        try:
            images_paths = []
            for file in files:
                if not file.content_type or not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400,detail="uploaded file isn't image")
                
                file_path = os.path.join(DOWNLOADED_IMAGES_PATHS,os.path.basename(file.filename))
                ensure_directories()
                with open(file_path,'wb') as f:
                    f.write(await file.read())
                images_paths.append(file_path)

            predictions = classifier.predict_batch(images_paths)
            delete_files(images_paths)
            return predictions
        except HTTPException:
            raise
        except Exception as e:
            logging.error(f"error in disk prediction endpoint {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  
        port=8001,
        reload=True,  
        log_level="info"
    )
