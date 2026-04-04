"""
Module de détection utilisant MobileViT entraîné
"""

import torch
from torchvision import transforms
from PIL import Image
from mobilevit_trainer import MobileViTClassifier
import os


class MobileViTDetector:
    """
    Détecteur d'images IA utilisant MobileViT
    """
    def __init__(self, model_path='models/mobilevit_best.pth', device=None):
        """
        Args:
            model_path: Chemin vers le modèle entraîné
            device: Device à utiliser (cuda/cpu)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"🔧 Chargement du modèle MobileViT sur {self.device}...")
        
        # Charger le modèle
        self.model = MobileViTClassifier(pretrained=False)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Modèle chargé depuis {model_path}")
        else:
            print(f"⚠️  Modèle non trouvé : {model_path}")
            print("   Utilisation du modèle pré-entraîné (non fine-tuné)")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transformations
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """
        Prédit si une image est générée par IA
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            dict avec score et probabilités
        """
        # Charger et transformer l'image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Probabilité que ce soit une image IA
        ai_probability = probabilities[0][1].item()
        
        return {
            "is_ai": predicted_class == 1,
            "ai_probability": ai_probability,
            "real_probability": probabilities[0][0].item(),
            "confidence": confidence,
            "predicted_class": "AI" if predicted_class == 1 else "Real"
        }
    
    def predict_batch(self, image_paths):
        """
        Prédit pour plusieurs images
        
        Args:
            image_paths: Liste de chemins d'images
            
        Returns:
            Liste de résultats
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result["image_path"] = image_path
                results.append(result)
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "error": str(e)
                })
        
        return results


# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mobilevit_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Créer le détecteur
    detector = MobileViTDetector()
    
    # Prédire
    result = detector.predict(image_path)
    
    print("\n" + "=" * 60)
    print("📊 RÉSULTAT DE LA DÉTECTION")
    print("=" * 60)
    print(f"\n🖼️  Image : {image_path}")
    print(f"\n🎯 Prédiction : {result['predicted_class']}")
    print(f"📈 Probabilité IA : {result['ai_probability']:.2%}")
    print(f"📈 Probabilité Réelle : {result['real_probability']:.2%}")
    print(f"🔍 Confiance : {result['confidence']:.2%}")
    print("=" * 60)
