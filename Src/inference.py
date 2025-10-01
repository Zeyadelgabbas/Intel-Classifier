from .config import MODEL,IDX2LABEL , TARGET_SIZE
from .logger import get_logger
import tensorflow as tf
import numpy as np 
from typing import Union
from .schemas import PredictionResponse, PredictionsResponse
import io
import os 
logging = get_logger(__name__)




class ImageClassifier:

    def __init__(self,model=MODEL,idx2label=IDX2LABEL,target_size =TARGET_SIZE):
        self.model = model
        self.idx2label = idx2label
        self.target_size = target_size

    def preprocess_image(self,image:Union[str,bytes]):

        try: 
            if isinstance(image,str):   # image is path 
                image = tf.keras.preprocessing.image.load_img(
                    image,target_size = self.target_size
                )
            else:
                image = tf.keras.preprocessing.image.load_img(
                    io.BytesIO(image) , target_size = self.target_size
                )

            image = tf.keras.preprocessing.image.img_to_array(image)
            image_array = np.squeeze(image)
            image_array = np.expand_dims(image,axis =0)
            image_array = tf.keras.applications.mobilenet_v2.preprocess_input(image_array)
            return image_array
        except Exception as e :
            logging.error("error in preprocessing image")
            raise e


    def predict_batch(self,images : Union[str,bytes]):

        if not images:
            logging.error("no images uploaded")
            raise ValueError("No images uploaded")
        
        basenames=[]
        for i , image in enumerate(images):
            if isinstance(image,str):
                basenames.append(os.path.basename(image))
            else:
                basenames.append(f"image_{i}")

        preprocessed_images = []
        for image in images:
            img_array=preprocessed_images.append(self.preprocess_image(image))

        batch = np.vstack(preprocessed_images)
        predictions= self.model.predict(batch,verbose = 0 )
        predicted_class_indices = np.argmax(predictions,axis = -1)
        predicted_class_names = [self.idx2label[i] for i in predicted_class_indices]
        confidence_score=[predictions[i][class_] 
                          for i , class_ in enumerate(predicted_class_indices)]
        
        prediction_responses = [PredictionResponse(
            base_name = basename,
            class_index=idx,
            class_name=name,
            confidence=conf
        ) for basename,idx,name,conf in zip(
            basenames , predicted_class_indices , predicted_class_names , confidence_score
        )]

        return PredictionsResponse(predictions=prediction_responses)


            

