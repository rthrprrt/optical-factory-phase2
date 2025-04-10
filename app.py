import cv2
import requests
import numpy as np
from io import BytesIO

# Adresse de votre API backend
API_URL = "http://localhost:8000/api/v1"

def main():
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    # Liste des modèles de lunettes
    glasses_models = ["sunglass_model_1", "sunglass_model_2", "sunglass_model_3"]
    current_model_index = 0
    
    # Analyser au démarrage pour obtenir la forme du visage
    face_shape = None
    recommendations = []
    
    print("Appuyez sur 'a' pour analyser votre visage")
    print("Appuyez sur 'n' pour passer au modèle suivant")
    print("Appuyez sur 'q' pour quitter")
    
    while True:
        # Capturer une image de la webcam
        ret, frame = cap.read()
        if not ret:
            break
        
        # Afficher le miroir (inversé horizontalement pour un effet miroir)
        mirror_frame = cv2.flip(frame, 1)
        
        # Si nous avons un modèle sélectionné, envoyer l'image pour superposition
        if face_shape:
            # Convertir l'image en bytes
            _, img_encoded = cv2.imencode('.jpg', frame)
            
            # Envoyer à l'API pour superposition
            files = {'image_file': ('webcam.jpg', img_encoded.tobytes(), 'image/jpeg')}
            data = {'glasses_id': glasses_models[current_model_index]}
            
            try:
                response = requests.post(f"{API_URL}/render_glasses", 
                                       files=files, 
                                       data=data, 
                                       timeout=1.0)  # Timeout court pour ne pas bloquer l'interface
                
                if response.status_code == 200:
                    # Convertir la réponse en image OpenCV
                    result_img = cv2.imdecode(np.frombuffer(response.content, np.uint8), 1)
                    mirror_frame = cv2.flip(result_img, 1)  # Inverser pour effet miroir
            except requests.exceptions.RequestException:
                # Ignorer les erreurs de timeout
                pass
        
        # Afficher les informations sur l'écran
        if face_shape:
            cv2.putText(mirror_frame, f"Forme: {face_shape}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(mirror_frame, f"Lunettes: {glasses_models[current_model_index]}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(mirror_frame, "Appuyez sur 'a' pour analyser", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Afficher l'image
        cv2.imshow('Essayage Virtuel de Lunettes', mirror_frame)
        
        # Traiter les touches
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Analyser le visage
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'image_file': ('webcam.jpg', img_encoded.tobytes(), 'image/jpeg')}
            
            try:
                response = requests.post(f"{API_URL}/analyze_and_recommend", 
                                        files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    if result['analysis']['detection_successful']:
                        face_shape = result['analysis']['detected_face_shape']
                        if result['recommendation']:
                            recommendations = result['recommendation']['recommended_glasses_ids']
                            glasses_models = recommendations
                            current_model_index = 0
            except requests.exceptions.RequestException as e:
                print(f"Erreur lors de l'analyse: {e}")
        
        elif key == ord('n'):
            # Passer au modèle suivant
            if face_shape and recommendations:
                current_model_index = (current_model_index + 1) % len(glasses_models)
    
    # Nettoyer
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()