#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple du système hybride
"""
import sys
import io
from pathlib import Path
from hybrid_detector import HybridAIDetector

# Forcer UTF-8 pour Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_simple():
    """Test rapide sur quelques images"""
    
    print("="*80)
    print("🧪 TEST SIMPLE DU SYSTÈME HYBRIDE")
    print("="*80)
    
    # Initialiser
    detector = HybridAIDetector()
    
    # Images de test
    test_images = [
        ("dataset/ai/-yellow-kimono-with-a-golden-crown-and-a-very-detailed-Barn-owl-copy-800x800.jpg", "AI", "> 0.6"),
        ("dataset/ai/00urban_biodiversity_landscape_architecture_14ed209d-45ca-4c57-b995-4d1552813cc1.jpg", "AI", "> 0.6"),
        ("dataset/reel/-man-sits-with-a-woman-on-her-phone-at-a-table-while-looking-at-a-computer_l.jpg", "Real", "< 0.4"),
    ]
    
    results = []
    
    for img_path, expected, threshold in test_images:
        print(f"\n{'='*80}")
        print(f"📸 Test : {Path(img_path).name}")
        print(f"   Attendu : {expected} (score {threshold})")
        print('='*80)
        
        try:
            result = detector.detect(img_path, save_results=False)
            
            if result.get("success"):
                score = result["scores"]["final_score"]
                llm = result["scores"]["llm_score"]
                mvit = result["scores"]["mobilevit_score"]
                
                # Vérifier si correct
                if expected == "AI":
                    correct = score > 0.6
                else:
                    correct = score < 0.4
                
                status = "✅ CORRECT" if correct else "❌ ERREUR"
                
                results.append({
                    "image": Path(img_path).name,
                    "expected": expected,
                    "score": score,
                    "correct": correct
                })
                
                print(f"\n{status}")
                print(f"Score final: {score:.2f}")
                print(f"  - LLM: {llm:.2f}")
                print(f"  - MobileViT: {mvit:.2f}")
            else:
                print(f"\n❌ Erreur lors de l'analyse")
                results.append({
                    "image": Path(img_path).name,
                    "expected": expected,
                    "error": True
                })
                
        except Exception as e:
            print(f"\n❌ Exception: {str(e)}")
            results.append({
                "image": Path(img_path).name,
                "expected": expected,
                "error": True
            })
    
    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ")
    print("="*80)
    
    correct = sum(1 for r in results if r.get("correct", False))
    total = len([r for r in results if not r.get("error", False)])
    
    print(f"\nTests réussis : {correct}/{total}")
    
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"Précision : {accuracy:.1f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    test_simple()
