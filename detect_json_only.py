#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détection d'images IA - Sortie JSON pure (sans messages)
Usage: python detect_json_only.py <chemin_image> [output.json]
"""
import sys
import json
import os
from pathlib import Path

# Supprimer tous les messages de progression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

from hybrid_detector import HybridAIDetector

def detect_silent(image_path, output_path=None):
    """
    Détecte une image et retourne/sauvegarde le JSON
    
    Args:
        image_path: Chemin de l'image
        output_path: Chemin du fichier JSON de sortie (optionnel)
    """
    # Vérifier l'image
    if not Path(image_path).exists():
        result = {
            "success": False,
            "error": f"Image non trouvée: {image_path}"
        }
    else:
        # Rediriger stdout temporairement pour supprimer les messages
        import io
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Analyser
            detector = HybridAIDetector(use_mobilevit=True)
            result = detector.detect(image_path, save_results=False)
        except Exception as e:
            result = {
                "success": False,
                "error": str(e)
            }
        finally:
            # Restaurer stdout
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    # Déterminer le fichier de sortie
    if output_path is None:
        image_name = Path(image_path).stem
        output_path = f"result_{image_name}.json"
    
    # Sauvegarder
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Afficher uniquement le chemin du fichier JSON
    print(output_path)
    
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_json_only.py <image> [output.json]", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    detect_silent(image_path, output_path)
