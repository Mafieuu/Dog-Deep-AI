from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import gdown
import os

app = Flask(__name__)

# Classes des races de chiens (dans l'ordre du modèle)
BREED_CLASSES = [
    "Afghan_hound",
    "African_hunting_dog", 
    "Airedale",
    "American_Staffordshire_terrier",
    "Appenzeller",
    "Australian_terrier",
    "Bedlington_terrier",
    "Bernese_mountain_dog",
    "Blenheim_spaniel",
    "Border_collie"
]

# Configuration globale
MODEL_PATH = "mobilenetv2_improved.h5"
GOOGLE_DRIVE_FILE_ID = "1_ALk5HD5ofWOfE_v99lfghZSy7zBgQAY"  
# Variable globale pour le modèle
model = None

def download_model_from_drive(file_id, output_path):
    """Télécharge le modèle depuis Google Drive"""
    gdown.download(id=file_id, output=output_path, quiet=False)
    #url = f"https://drive.google.com/uc?id={file_id}&export=download"
    gdown.download(id=file_id, output=output_path, quiet=False)
   


def load_model():
    """Charge le modèle TensorFlow"""
    global model
    if not os.path.exists(MODEL_PATH):
        print(f"Téléchargement du modèle depuis Google Drive...")
        download_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modèle chargé avec succès!")

def preprocess_image(image_path, img_size=224, annotation_path=None):
    """
    Pipeline de préprocessing d'image pour les modèles Stanford Dogs
    
    Args:
        image_path (str): Chemin vers l'image
        img_size (int): Taille de sortie (224 par défaut)
        annotation_path (str, optional): Chemin vers le fichier d'annotation XML
    
    Returns:
        tf.Tensor: Image preprocessée de forme (1, img_size, img_size, 3)
    """
    # Charger l'image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, tf.float32)
    
    # Cropper l'image si annotation fournie
    if annotation_path is not None and os.path.exists(annotation_path):
        try:
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            
            # Trouver le premier objet avec bounding box
            obj = root.find('object')
            if obj is None:
                bbox = root.find('bndbox')
            else:
                bbox = obj.find('bndbox')
            
            if bbox is not None:
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                
                # Cropper l'image selon la bounding box
                image = tf.image.crop_to_bounding_box(
                    image,
                    offset_height=ymin,
                    offset_width=xmin,
                    target_height=ymax - ymin,
                    target_width=xmax - xmin
                )
        except Exception as e:
            print(f"Erreur lors du parsing XML: {e}")
    
    # Redimensionner
    image = tf.image.resize(image, [img_size, img_size])
    
    # Normaliser
    image = image / 255.0
    
    # Ajouter dimension batch
    image = tf.expand_dims(image, 0)
    
    return image

def get_top_predictions(predictions, top_k=3):
    """
    Retourne les top-k prédictions avec noms et probabilités
    
    Args:
        predictions: Array numpy des prédictions
        top_k: Nombre de prédictions à retourner
    
    Returns:
        list: Liste des dictionnaires {breed, probability}
    """
    # Obtenir les indices des top-k prédictions
    top_indices = np.argsort(predictions[0])[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "breed": BREED_CLASSES[idx],
            "probability": float(predictions[0][idx])
        })
    
    return results

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint de prédiction
    
    Expected JSON:
    {
        "image_path": "/path/to/image.jpg",
        "annotation_path": "/path/to/annotation.xml" (optionnel)
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image_path' not in data:
            return jsonify({"error": "image_path requis"}), 400
        
        image_path = data['image_path']
        annotation_path = data.get('annotation_path', None)
        
        # Vérifier si l'image existe
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image non trouvée: {image_path}"}), 404
        
        # Préprocesser l'image
        processed_image = preprocess_image(image_path, 224, annotation_path)
        
        # Faire la prédiction
        predictions = model.predict(processed_image)
        
        # Obtenir les top 3 prédictions
        top_predictions = get_top_predictions(predictions, top_k=3)
        
        return jsonify({
            "success": True,
            "predictions": top_predictions
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Endpoint de santé"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route('/breeds', methods=['GET'])
def get_breeds():
    """Retourne la liste des races supportées"""
    return jsonify({
        "breeds": BREED_CLASSES,
        "total": len(BREED_CLASSES)
    })

if __name__ == '__main__':
    # Charger le modèle au démarrage
    load_model()
    
    # Lancer l'API
    app.run(debug=True, host='0.0.0.0', port=5000)