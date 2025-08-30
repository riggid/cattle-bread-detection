import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import json
import os

# --- GPU MEMORY CONFIGURATION ---
def configure_gpu():
    """Configure GPU to prevent memory issues"""
    try:
        # Clear any existing TensorFlow sessions first
        tf.keras.backend.clear_session()
        
        # Get GPU devices
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            print(f"🔍 Found {len(gpus)} GPU(s)")
            
            # Reset GPU memory stats
            try:
                tf.config.experimental.reset_memory_stats(gpus[0])
                print("🧹 GPU memory stats reset")
            except Exception as reset_error:
                print(f"⚠️ Could not reset GPU memory: {reset_error}")
            
            # Configure memory growth
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print("✅ GPU memory growth enabled")
            except RuntimeError as growth_error:
                print(f"⚠️ Memory growth config failed: {growth_error}")
                # If memory growth fails, try limiting memory instead
                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
                    )
                    print("✅ GPU memory limited to 2GB")
                except Exception as limit_error:
                    print(f"❌ GPU configuration completely failed: {limit_error}")
                    return False
            
            # Set soft device placement
            tf.config.set_soft_device_placement(True)
            return True
            
        else:
            print("⚠️ No GPUs found, will use CPU")
            return False
            
    except Exception as e:
        print(f"❌ GPU configuration error: {e}")
        return False

# Configure GPU before loading anything
gpu_available = configure_gpu()

# Force garbage collection
import gc
gc.collect()

# --- CONFIGURATION ---
MODEL_PATH = "models/cattle_cnn_rgb.keras"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)

# --- LOAD MODEL & CLASS NAMES ---
def load_model_and_classes():
    """Load the trained model and class names with GPU fallback"""
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        print("🔄 Loading model...")
        
        # Try loading with different strategies
        model = None
        
        # Strategy 1: Try GPU with memory growth
        if gpu_available:
            try:
                print("🎮 Attempting GPU loading...")
                model = tf.keras.models.load_model(MODEL_PATH)
                print("✅ Model loaded successfully on GPU")
            except Exception as gpu_error:
                print(f"⚠️ GPU loading failed: {gpu_error}")
                model = None
        
        # Strategy 2: Fallback to CPU if GPU fails
        if model is None:
            print("🔄 Falling back to CPU loading...")
            # Force CPU loading
            with tf.device('/CPU:0'):
                model = tf.keras.models.load_model(MODEL_PATH)
            print("✅ Model loaded successfully on CPU")
        
        # Load or create class names
        if os.path.exists(CLASS_NAMES_PATH):
            with open(CLASS_NAMES_PATH, "r") as f:
                class_names = json.load(f)
            print(f"✅ Class names loaded from {CLASS_NAMES_PATH}")
        else:
            # Default class names from your training
            class_names = [
                'Ayrshire cattle', 
                'Brown Swiss cattle', 
                'Holstein Friesian cattle', 
                'Jersey cattle', 
                'Red Dane cattle'
            ]
            # Save for future use
            os.makedirs("models", exist_ok=True)
            with open(CLASS_NAMES_PATH, "w") as f:
                json.dump(class_names, f)
            print(f"✅ Class names created and saved to {CLASS_NAMES_PATH}")
        
        print(f"📊 Model expects {len(class_names)} classes: {class_names}")
        return model, class_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# Load model and class names once
model, CLASS_NAMES = load_model_and_classes()

def predict_breed(img_path):
    """
    Predict cattle breed from image with robust error handling
    
    Args:
        img_path (str): Path to the image file
    
    Returns:
        tuple: (breed_name, confidence) or (None, 0.0) if failed
    """
    print(f"🔍 Starting prediction for: {img_path}")
    
    if model is None or CLASS_NAMES is None:
        print("❌ Model or class names not loaded")
        return None, 0.0
    
    try:
        # Validate image path
        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            return None, 0.0
        
        print(f"📁 Image file size: {os.path.getsize(img_path)} bytes")
        
        # Load and preprocess image
        print("🖼️ Loading image...")
        img = Image.open(img_path)
        print(f"📐 Original image size: {img.size}, mode: {img.mode}")
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
            print("🎨 Converted to RGB")
        
        # Resize to model input size
        img = img.resize(IMG_SIZE)
        print(f"📏 Resized to: {IMG_SIZE}")
        
        # Convert to array and normalize (exactly like training)
        img_array = np.array(img, dtype=np.float32)
        print(f"📊 Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
        print(f"📊 Image array range: {img_array.min():.3f} to {img_array.max():.3f}")
        
        # Normalize
        img_array = img_array / 255.0
        print(f"📊 After normalization: {img_array.min():.3f} to {img_array.max():.3f}")
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        print(f"📊 Final shape for prediction: {img_array.shape}")
        
        # Make prediction
        print("🤖 Running prediction...")
        
        # Try GPU first, fallback to CPU
        preds = None
        device_used = "Unknown"
        
        if gpu_available:
            try:
                with tf.device('/GPU:0'):
                    preds = model.predict(img_array, batch_size=1, verbose=0)
                device_used = "GPU"
                print("✅ Prediction completed on GPU")
            except Exception as gpu_error:
                print(f"⚠️ GPU prediction failed: {gpu_error}")
                preds = None
        
        # CPU fallback
        if preds is None:
            try:
                with tf.device('/CPU:0'):
                    preds = model.predict(img_array, batch_size=1, verbose=0)
                device_used = "CPU"
                print("✅ Prediction completed on CPU")
            except Exception as cpu_error:
                print(f"❌ CPU prediction also failed: {cpu_error}")
                return None, 0.0
        
        # Process results
        print(f"📈 Prediction array shape: {preds.shape}")
        print(f"📈 Prediction values: {preds[0]}")
        
        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        breed = CLASS_NAMES[predicted_class]
        
        print(f"🎯 Predicted class index: {predicted_class}")
        print(f"🎯 Breed: {breed}")
        print(f"🎯 Confidence: {confidence:.4f}")
        print(f"🎯 Device used: {device_used}")
        
        return breed, confidence
        
    except Exception as e:
        print(f"❌ Error predicting breed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

def predict_breed_batch(img_paths):
    """
    Predict breeds for multiple images efficiently
    
    Args:
        img_paths (list): List of image file paths
    
    Returns:
        list: List of (breed, confidence) tuples
    """
    if model is None or CLASS_NAMES is None:
        return []
    
    try:
        # Process images in small batches to manage GPU memory
        batch_size = 4  # Small batch size to prevent memory issues
        results = []
        
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            batch_images = []
            
            # Load and preprocess batch
            for img_path in batch_paths:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img = img.resize(IMG_SIZE)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    batch_images.append(img_array)
            
            if batch_images:
                # Convert to batch tensor
                batch_tensor = np.array(batch_images)
                
                # Predict on GPU
                with tf.device('/GPU:0'):
                    preds = model.predict(batch_tensor, batch_size=len(batch_images), verbose=0)
                
                # Process results
                for j, pred in enumerate(preds):
                    predicted_class = np.argmax(pred)
                    confidence = float(np.max(pred))
                    breed = CLASS_NAMES[predicted_class]
                    results.append((breed, confidence))
            
            # Clear cache between batches
            tf.keras.backend.clear_session()
        
        return results
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {e}")
        return []

# --- MEMORY MONITORING ---
def check_gpu_memory():
    """Check current GPU memory usage"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            memory_info = tf.config.experimental.get_memory_info(gpus[0])
            current_mb = memory_info['current'] / 1024 / 1024
            peak_mb = memory_info['peak'] / 1024 / 1024
            print(f"📊 GPU Memory - Current: {current_mb:.1f}MB, Peak: {peak_mb:.1f}MB")
            return current_mb, peak_mb
    except Exception as e:
        print(f"⚠️ Could not check GPU memory: {e}")
    return 0, 0

# --- CLEANUP FUNCTION ---
def cleanup_gpu_memory():
    """Manually clean up GPU memory"""
    try:
        tf.keras.backend.clear_session()
        # Force garbage collection
        import gc
        gc.collect()
        print("🧹 GPU memory cleaned")
    except Exception as e:
        print(f"⚠️ GPU cleanup warning: {e}")

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Check GPU status
    print("🔍 GPU Status Check:")
    check_gpu_memory()
    
    # Test prediction
    test_image = "path/to/test/image.jpg"  # Replace with actual path
    
    if os.path.exists(test_image):
        print("\n🔍 Testing prediction...")
        breed, confidence = predict_breed(test_image)
        
        if breed:
            print(f"📸 Predicted breed: {breed}")
            print(f"🎯 Confidence: {confidence:.2%}")
        else:
            print("❌ Prediction failed")
        
        # Check memory usage after prediction
        check_gpu_memory()
    else:
        print("✅ Prediction functions loaded successfully!")
        print("Update the test_image path to test predictions.")