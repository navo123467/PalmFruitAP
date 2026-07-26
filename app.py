from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import io


app = FastAPI()


model = tf.keras.models.load_model(
    "best_palm_model.keras"
)


classes = [
    "Unripe",
    "Ripe",
    "Overripe"
]


class ImageRequest(BaseModel):
    image:str



@app.post("/predict")
async def predict(data:ImageRequest):

    image_data = data.image.split(",")[1]


    image_bytes = base64.b64decode(image_data)


    image = Image.open(
        io.BytesIO(image_bytes)
    )


    image = image.resize((224,224))


    image = np.array(image)/255.0


    image = np.expand_dims(
        image,
        axis=0
    )


    prediction = model.predict(image)


    index = np.argmax(prediction)


    confidence = float(
        prediction[0][index]*100
    )


    return {
        "class":classes[index],
        "confidence":round(confidence,2)
    }