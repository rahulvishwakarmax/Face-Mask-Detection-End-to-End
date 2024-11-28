from fastapi import FastAPI, File, UploadFile  
from fastapi.middleware.cors import CORSMiddleware  
import uvicorn  
import numpy as np  
from io import BytesIO  
from PIL import Image  
import tensorflow as tf  
import os  

app = FastAPI()  

origins = [  
    "http://localhost",  
    "http://localhost:3000",  
]  
app.add_middleware(  
    CORSMiddleware,  
    allow_origins=origins,  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)  

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/face_mask_model.keras'))  
MODEL = tf.keras.models.load_model(model_path)  

CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]  

IMAGE_SIZE = 128  

@app.get("/ping")  
async def ping():  
    return {"message": "Hello, I am alive"}  

def read_file_as_image(data) -> np.ndarray:  
    image = Image.open(BytesIO(data)).convert('RGB')  
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))  
    image = np.array(image) / 255.0   
    return image  

@app.post("/predict")  
async def predict(file: UploadFile = File(...)):  
    try:  
        image = read_file_as_image(await file.read())  
    except Exception as e:  
        return {"error": f"Error processing image: {str(e)}"}  
    
    img_batch = np.expand_dims(image, 0)  

    try:  
        predictions = MODEL.predict(img_batch)  
    except Exception as e:  
        return {"error": f"Error making prediction: {str(e)}"}  

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]  
    confidence = np.max(predictions[0])  
    return {  
        'class': predicted_class,  
        'confidence': float(confidence)  
    }  

if __name__ == "__main__":  
    uvicorn.run(app, host='localhost', port=8000, log_level="debug")