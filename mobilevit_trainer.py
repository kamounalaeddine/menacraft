"""
Module d'entraînement MobileViT pour la détection d'images IA
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import json


class AIImageDataset(Dataset):
    """
    Dataset pour les images IA et réelles
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Dossier racine (train/ ou val/)
            transform: Transformations à appliquer
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Charger les images réelles (label = 0)
        real_dir = self.root_dir / "real"
        if real_dir.exists():
            for img_path in real_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(str(img_path))
                    self.labels.append(0)  # 0 = réelle
        
        # Charger les images IA (label = 1)
        ai_dir = self.root_dir / "ai"
        if ai_dir.exists():
            for img_path in ai_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.images.append(str(img_path))
                    self.labels.append(1)  # 1 = IA
        
        print(f"📊 Dataset chargé : {len(self.images)} images")
        print(f"   - Réelles : {self.labels.count(0)}")
        print(f"   - IA : {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Charger l'image
        image = Image.open(img_path).convert('RGB')
        
        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label


class MobileViTClassifier(nn.Module):
    """
    Classificateur basé sur MobileViT
    """
    def __init__(self, pretrained=True):
        super(MobileViTClassifier, self).__init__()
        
        # Charger MobileViT pré-entraîné
        try:
            from transformers import MobileViTForImageClassification
            self.mobilevit = MobileViTForImageClassification.from_pretrained(
                "apple/mobilevit-small",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
        except ImportError:
            print("⚠️  transformers non installé. Installation requise :")
            print("   pip install transformers")
            raise
    
    def forward(self, x):
        outputs = self.mobilevit(x)
        return outputs.logits


class MobileViTTrainer:
    """
    Entraîneur pour le modèle MobileViT
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader):
        """
        Entraîne le modèle pour une époque
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Statistiques
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Mise à jour de la barre de progression
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """
        Valide le modèle
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.0001):
        """
        Entraîne le modèle
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"\n🚀 Début de l'entraînement sur {self.device}")
        print(f"   Époques : {epochs}")
        print(f"   Learning rate : {lr}")
        print()
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\n📊 Époque {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Entraînement
            train_loss, train_acc = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\n📈 Résultats Époque {epoch + 1}:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Sauvegarder le meilleur modèle
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('models/mobilevit_best.pth')
                print(f"   ✅ Meilleur modèle sauvegardé (Val Acc: {val_acc:.2f}%)")
        
        print(f"\n🎉 Entraînement terminé !")
        print(f"   Meilleure précision validation : {best_val_acc:.2f}%")
        
        return self.history
    
    def save_model(self, path):
        """
        Sauvegarde le modèle
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
    
    def load_model(self, path):
        """
        Charge le modèle
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', self.history)


def get_transforms():
    """
    Retourne les transformations pour l'entraînement et la validation
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def main():
    """
    Fonction principale d'entraînement
    """
    print("=" * 80)
    print("🧠 ENTRAÎNEMENT MOBILEVIT POUR DÉTECTION D'IMAGES IA")
    print("=" * 80)
    print()
    
    # Vérifier que le dataset existe
    if not os.path.exists("dataset/train"):
        print("❌ Erreur : Le dossier dataset/train n'existe pas")
        print("\n📁 Structure requise :")
        print("   dataset/")
        print("   ├── train/")
        print("   │   ├── real/    # Images réelles")
        print("   │   └── ai/      # Images IA")
        print("   └── val/")
        print("       ├── real/")
        print("       └── ai/")
        return
    
    # Préparer les transformations
    train_transform, val_transform = get_transforms()
    
    # Charger les datasets
    print("📂 Chargement des datasets...")
    train_dataset = AIImageDataset("dataset/train", transform=train_transform)
    val_dataset = AIImageDataset("dataset/val", transform=val_transform)
    
    if len(train_dataset) == 0:
        print("❌ Aucune image trouvée dans dataset/train")
        return
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Créer le modèle
    print("\n🏗️  Création du modèle MobileViT...")
    model = MobileViTClassifier(pretrained=True)
    
    # Créer le trainer
    trainer = MobileViTTrainer(model)
    
    # Entraîner
    history = trainer.train(train_loader, val_loader, epochs=10, lr=0.0001)
    
    # Sauvegarder l'historique
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n💾 Historique sauvegardé dans models/training_history.json")


if __name__ == "__main__":
    main()
