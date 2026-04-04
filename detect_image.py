#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de détection d'images IA - Sortie JSON uniquement
Usage: python detect_image.py <chemin_image>
"""
import sys
import json
from pathlib import Path
from hybrid_detector import HybridAIDetector

def detect_and_save_json(image_path):
    """
    Détecte si une image est IA et sauvegarde le résultat en JSON
    
    Args:
        image_path: Chemin vers l'image à analyser
    """
    # Vérifier que l'image existe
    if not Path(image_path).exists():
        error_result = {
            "success": False,
            "error": f"Image non trouvée: {image_path}"
        }
        output_file = "error_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, ensure_ascii=False, indent=2)
        print(f"❌ Erreur sauvegardée dans: {output_file}")
        return
    
    # Initialiser le détecteur
    print("🔧 Initialisation du détecteur...")
    detector = HybridAIDetector(use_mobilevit=True)
    
    # Analyser l'image
    print(f"\n🔍 Analyse de: {image_path}")
    result = detector.detect(image_path, save_results=False)
    
    # Créer le nom du fichier de sortie
    image_name = Path(image_path).stem
    output_file = f"result_{image_name}.json"
    
    # Sauvegarder le résultat
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # Afficher le résumé
    if result.get("success"):
        score = result["scores"]["final_score"]
        conclusion = result["conclusion"]
        
        print(f"\n{'='*80}")
        print("✅ ANALYSE TERMINÉE")
        print('='*80)
        print(f"📈 Score: {score:.2f}")
        print(f"💡 Conclusion: {conclusion}")
        print(f"💾 Résultat sauvegardé: {output_file}")
        print('='*80)
    else:
        print(f"\n❌ Erreur lors de l'analyse")
        print(f"💾 Résultat sauvegardé: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_image.py <chemin_image>")
        print("\nExemples:")
        print('  python detect_image.py "dataset/ai/image.jpg"')
        print('  python detect_image.py "C:/Users/fida/Desktop/photo.png"')
        print('  python detect_image.py "mon_image.jpg"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    detect_and_save_json(image_path)
