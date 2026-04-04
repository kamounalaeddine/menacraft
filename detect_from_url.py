#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Détecte une image directement depuis une URL
Usage: python detect_from_url.py <url_image>
"""
import sys
import io
import json
import requests
from pathlib import Path
from hybrid_detector import HybridAIDetector

# Forcer UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def download_image(url, filename):
    """Télécharge une image depuis une URL"""
    try:
        print(f"📥 Téléchargement de l'image...")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Image téléchargée: {filename}")
        return True
    except Exception as e:
        print(f"❌ Erreur téléchargement: {e}")
        return False

def detect_from_url(url):
    """Détecte une image depuis une URL"""
    
    print("="*80)
    print("🌐 DÉTECTION D'IMAGE DEPUIS URL")
    print("="*80)
    print(f"\n🔗 URL: {url}\n")
    
    # Créer dossier temporaire
    temp_dir = Path("temp_downloads")
    temp_dir.mkdir(exist_ok=True)
    
    # Nom du fichier temporaire
    import hashlib
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    temp_file = temp_dir / f"image_{url_hash}.jpg"
    
    # Télécharger
    if not download_image(url, temp_file):
        return
    
    # Initialiser le détecteur
    print("\n🔧 Initialisation du détecteur...")
    detector = HybridAIDetector()
    
    # Analyser
    print(f"\n🔍 Analyse de l'image...\n")
    result = detector.detect(str(temp_file), save_results=False)
    
    if result.get("success"):
        score = result["scores"]["final_score"]
        llm = result["scores"]["llm_score"]
        mvit = result["scores"]["mobilevit_score"]
        conclusion = result["conclusion"]
        
        print(f"\n{'='*80}")
        print("📊 RÉSULTAT")
        print('='*80)
        print(f"\n📈 Score final: {score:.2f}")
        print(f"💡 Conclusion: {conclusion}")
        print(f"\n📊 Détails:")
        print(f"   - LLM (60%): {llm:.2f}")
        print(f"   - MobileViT (40%): {mvit:.2f}")
        
        # Interprétation
        print(f"\n🎯 Interprétation:")
        if score < 0.3:
            print("   → Très probablement une photo réelle")
        elif score < 0.5:
            print("   → Probablement une photo réelle")
        elif score < 0.7:
            print("   → Incertain, nécessite vérification manuelle")
        elif score < 0.85:
            print("   → Probablement générée par IA")
        else:
            print("   → Très probablement générée par IA")
        
        print('='*80)
        
        # Sauvegarder le résultat
        output_file = f"result_url_{url_hash}.json"
        result["source_url"] = url
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Résultat sauvegardé: {output_file}")
        print(f"📁 Image téléchargée: {temp_file}")
    else:
        print(f"\n❌ Erreur lors de l'analyse")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_from_url.py <url_image>")
        print("\nExemples:")
        print('  python detect_from_url.py "https://example.com/image.jpg"')
        print('  python detect_from_url.py "https://i.imgur.com/abc123.png"')
        sys.exit(1)
    
    url = sys.argv[1]
    detect_from_url(url)
