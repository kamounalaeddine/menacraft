"""
Script pour préparer le dataset à partir de dataset/ai et dataset/reel
Divise automatiquement en train (80%) et val (20%)
"""

import os
import shutil
from pathlib import Path
import random


def prepare_dataset(source_dir="dataset", train_ratio=0.8):
    """
    Prépare le dataset en train/val à partir de dataset/ai et dataset/reel
    
    Args:
        source_dir: Dossier source contenant ai/ et reel/
        train_ratio: Ratio pour l'entraînement (0.8 = 80% train, 20% val)
    """
    print("=" * 80)
    print("📁 PRÉPARATION DU DATASET")
    print("=" * 80)
    print()
    
    source_path = Path(source_dir)
    
    # Vérifier que les dossiers source existent
    ai_source = source_path / "ai"
    reel_source = source_path / "reel"
    
    if not ai_source.exists():
        print(f"❌ Erreur : {ai_source} n'existe pas")
        return False
    
    if not reel_source.exists():
        print(f"❌ Erreur : {reel_source} n'existe pas")
        return False
    
    # Créer la structure train/val
    train_dir = source_path / "train"
    val_dir = source_path / "val"
    
    # Créer les dossiers
    (train_dir / "ai").mkdir(parents=True, exist_ok=True)
    (train_dir / "real").mkdir(parents=True, exist_ok=True)
    (val_dir / "ai").mkdir(parents=True, exist_ok=True)
    (val_dir / "real").mkdir(parents=True, exist_ok=True)
    
    print("✅ Structure de dossiers créée")
    print()
    
    # Fonction pour copier les images
    def split_and_copy(source, train_dest, val_dest, label):
        # Lister toutes les images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']
        images = []
        for ext in extensions:
            images.extend(list(source.glob(ext)))
            images.extend(list(source.glob(ext.upper())))
        
        if len(images) == 0:
            print(f"⚠️  Aucune image trouvée dans {source}")
            return 0, 0
        
        # Mélanger aléatoirement
        random.shuffle(images)
        
        # Calculer le split
        n_train = int(len(images) * train_ratio)
        train_images = images[:n_train]
        val_images = images[n_train:]
        
        # Copier les images d'entraînement
        for img in train_images:
            shutil.copy2(img, train_dest / img.name)
        
        # Copier les images de validation
        for img in val_images:
            shutil.copy2(img, val_dest / img.name)
        
        print(f"📊 {label} :")
        print(f"   Total : {len(images)} images")
        print(f"   Train : {len(train_images)} images → {train_dest}")
        print(f"   Val   : {len(val_images)} images → {val_dest}")
        print()
        
        return len(train_images), len(val_images)
    
    # Traiter les images IA
    print("🔴 Traitement des images IA...")
    ai_train, ai_val = split_and_copy(
        ai_source,
        train_dir / "ai",
        val_dir / "ai",
        "Images IA"
    )
    
    # Traiter les images réelles
    print("🟢 Traitement des images réelles...")
    real_train, real_val = split_and_copy(
        reel_source,
        train_dir / "real",
        val_dir / "real",
        "Images Réelles"
    )
    
    # Résumé
    print("=" * 80)
    print("📊 RÉSUMÉ")
    print("=" * 80)
    print()
    print(f"✅ Dataset préparé avec succès !")
    print()
    print(f"📁 Train : {ai_train + real_train} images")
    print(f"   - IA : {ai_train}")
    print(f"   - Réelles : {real_train}")
    print()
    print(f"📁 Validation : {ai_val + real_val} images")
    print(f"   - IA : {ai_val}")
    print(f"   - Réelles : {real_val}")
    print()
    
    # Vérifier si suffisant
    total = ai_train + ai_val + real_train + real_val
    if total < 20:
        print("⚠️  ATTENTION : Peu d'images (< 20)")
        print("   Recommandation : Ajouter plus d'images pour un meilleur entraînement")
    elif total < 50:
        print("⚠️  Dataset petit (< 50 images)")
        print("   Recommandation : Ajouter plus d'images si possible")
    else:
        print("✅ Dataset de bonne taille pour l'entraînement")
    
    print()
    print("🚀 Prochaine étape : python mobilevit_trainer.py")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    import sys
    
    # Ratio par défaut : 80% train, 20% val
    train_ratio = 0.8
    
    if len(sys.argv) > 1:
        try:
            train_ratio = float(sys.argv[1])
            if not 0 < train_ratio < 1:
                print("❌ Le ratio doit être entre 0 et 1")
                sys.exit(1)
        except ValueError:
            print("❌ Ratio invalide. Utilisation du ratio par défaut (0.8)")
            train_ratio = 0.8
    
    print(f"📊 Ratio train/val : {train_ratio:.0%} / {(1-train_ratio):.0%}")
    print()
    
    success = prepare_dataset(train_ratio=train_ratio)
    
    if not success:
        sys.exit(1)
