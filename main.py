from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Plant Disease Detection API",
    description="API for detecting plant diseases from uploaded images using CNN model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Class names for plant diseases
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Global variable to store the model
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        # Update this path to where your model is saved
        model_path = "plant_disease_model.keras"
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the uploaded image for prediction"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size (128x128)
        image = image.resize((128, 128))
        
        # Convert to array and normalize
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.array([img_array])  # Create batch dimension
        
        # Normalize pixel values to [0, 1] if needed
        # img_array = img_array / 255.0  # Uncomment if your model was trained with normalized images
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts"""
    load_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Plant Disease Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict plant disease from uploaded image
    
    Args:
        file: Uploaded image file
    
    Returns:
        Dictionary containing prediction results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an image (JPEG, PNG, etc.)"
            )
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(
                status_code=500, 
                detail="Model not loaded. Please try again later."
            )
        
        # Read and process the image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Get the predicted class
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[0][predicted_class_index])
        
        # Get top 3 predictions
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                "class": CLASS_NAMES[i],
                "confidence": float(predictions[0][i]),
                "percentage": f"{float(predictions[0][i]) * 100:.2f}%"
            }
            for i in top_3_indices
        ]
        
        # Parse the predicted class name
        if "___" in predicted_class:
            plant_type, disease = predicted_class.split("___", 1)
            plant_type = plant_type.replace("_", " ").title()
            disease = disease.replace("_", " ").title()
            if disease.lower() == "healthy":
                status = "Healthy"
                disease = "No disease detected"
            else:
                status = "Diseased"
        else:
            plant_type = predicted_class.replace("_", " ").title()
            disease = "Unknown"
            status = "Unknown"
        
        return {
            "success": True,
            "prediction": {
                "plant_type": plant_type,
                "disease": disease,
                "status": status,
                "confidence": confidence,
                "confidence_percentage": f"{confidence * 100:.2f}%"
            },
            "top_predictions": top_3_predictions,
            "raw_prediction": predicted_class
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/batch_predict")
async def batch_predict_diseases(files: list[UploadFile] = File(...)) -> Dict[str, Any]:
    """
    Predict plant diseases from multiple uploaded images
    
    Args:
        files: List of uploaded image files
    
    Returns:
        Dictionary containing batch prediction results
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400, 
                detail="Maximum 10 images allowed per batch"
            )
        
        results = []
        for i, file in enumerate(files):
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "error": "File must be an image"
                    })
                    continue
                
                # Process single image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                processed_image = preprocess_image(image)
                
                # Make prediction
                predictions = model.predict(processed_image)
                predicted_class_index = np.argmax(predictions[0])
                predicted_class = CLASS_NAMES[predicted_class_index]
                confidence = float(predictions[0][predicted_class_index])
                
                # Parse result
                if "___" in predicted_class:
                    plant_type, disease = predicted_class.split("___", 1)
                    plant_type = plant_type.replace("_", " ").title()
                    disease = disease.replace("_", " ").title()
                    status = "Healthy" if disease.lower() == "healthy" else "Diseased"
                else:
                    plant_type = predicted_class.replace("_", " ").title()
                    disease = "Unknown"
                    status = "Unknown"
                
                results.append({
                    "filename": file.filename,
                    "plant_type": plant_type,
                    "disease": disease,
                    "status": status,
                    "confidence": confidence,
                    "confidence_percentage": f"{confidence * 100:.2f}%"
                })
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "total_images": len(files),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/classes")
async def get_classes():
    """Get all available plant disease classes"""
    return {
        "success": True,
        "total_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Change "main" to your filename if different
        host="127.0.0.1",
        port=8000,
        reload=True  # Set to False in production
    )