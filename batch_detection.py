"""
Script pour analyser plusieurs images en batch
"""

from ai_image_detector import AIImageDetector
from config import LLAVA_API_CONFIG, LLAMA_API_CONFIG
import os
import json
from pathlib import Path
import sys


def analyze_batch(image_folder: str, output_folder: str = "batch_results"):
    """
    Analyse toutes les images d'un dossier
    
    Args:
        image_folder: Dossier contenant les images
        output_folder: Dossier pour les résultats
    """
    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
    
    # Trouver toutes les images
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_folder).glob(f"*{ext}"))
        image_files.extend(Path(image_folder).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ Aucune image trouvée dans : {image_folder}")
        return
    
    print(f"📁 {len(image_files)} image(s) trouvée(s)\n")
    
    # Créer le dossier de sortie
    os.makedirs(output_folder, exist_ok=True)
    
    # Créer le détecteur
    print("🔧 Initialisation du détecteur...\n")
    detector = AIImageDetector(
        llava_api_key=LLAVA_API_CONFIG["api_key"],
        llava_base_url=LLAVA_API_CONFIG["base_url"],
        llama_api_key=LLAMA_API_CONFIG["api_key"],
        llama_base_url=LLAMA_API_CONFIG["base_url"],
        llava_model=LLAVA_API_CONFIG["model"],
        llama_model=LLAMA_API_CONFIG["model"]
    )
    
    # Résultats globaux
    all_results = []
    stats = {
        "total": len(image_files),
        "success": 0,
        "failed": 0,
        "ai_generated": 0,
        "real": 0,
        "uncertain": 0
    }
    
    # Analyser chaque image
    for i, image_path in enumerate(image_files, 1):
        print("=" * 80)
        print(f"📸 Image {i}/{len(image_files)} : {image_path.name}")
        print("=" * 80)
        
        try:
            result = detector.detect(str(image_path), save_results=False)
            
            if result.get("success"):
                stats["success"] += 1
                
                # Classifier le résultat
                score = result["score_global"]
                if score < 0.4:
                    stats["real"] += 1
                    category = "RÉELLE"
                elif score < 0.6:
                    stats["uncertain"] += 1
                    category = "INCERTAIN"
                else:
                    stats["ai_generated"] += 1
                    category = "IA"
                
                print(f"\n✅ {category} - Score: {score:.2f} - {result['conclusion']}\n")
                
                # Sauvegarder le résultat individuel
                output_file = Path(output_folder) / f"{image_path.stem}_result.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                all_results.append({
                    "image": image_path.name,
                    "score": score,
                    "conclusion": result["conclusion"],
                    "category": category
                })
            else:
                stats["failed"] += 1
                print(f"\n❌ Échec : {result.get('error')}\n")
                
        except Exception as e:
            stats["failed"] += 1
            print(f"\n❌ Erreur : {e}\n")
    
    # Sauvegarder le rapport global
    summary = {
        "statistics": stats,
        "results": all_results
    }
    
    summary_file = Path(output_folder) / "batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Afficher le résumé
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ DE L'ANALYSE BATCH")
    print("=" * 80)
    print(f"\n✅ Analysées avec succès : {stats['success']}/{stats['total']}")
    print(f"❌ Échecs : {stats['failed']}/{stats['total']}")
    print(f"\n📈 Classification :")
    print(f"   🟢 Images réelles : {stats['real']}")
    print(f"   🟡 Incertaines : {stats['uncertain']}")
    print(f"   🔴 Générées par IA : {stats['ai_generated']}")
    print(f"\n💾 Résultats sauvegardés dans : {output_folder}/")
    print(f"📄 Rapport global : {summary_file}")
    print("=" * 80)


def main():
    """
    Point d'entrée du script
    """
    print("=" * 80)
    print("🤖 ANALYSE BATCH D'IMAGES - DÉTECTION IA")
    print("=" * 80)
    print()
    
    # Récupérer le dossier d'images
    if len(sys.argv) > 1:
        image_folder = sys.argv[1]
    else:
        print("📝 Aucun dossier spécifié en argument")
        image_folder = input("   Chemin du dossier contenant les images : ").strip()
    
    if not image_folder:
        print("\n⚠️  Aucun dossier fourni. Analyse annulée.")
        print("\n💡 Usage : python batch_detection.py <dossier_images>")
        sys.exit(0)
    
    if not os.path.isdir(image_folder):
        print(f"\n❌ Erreur : Dossier non trouvé : {image_folder}")
        sys.exit(1)
    
    # Dossier de sortie
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "batch_results"
    
    # Lancer l'analyse
    try:
        analyze_batch(image_folder, output_folder)
    except KeyboardInterrupt:
        print("\n\n⚠️  Analyse interrompue par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur inattendue : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
