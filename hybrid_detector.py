"""
Détecteur hybride combinant LLaVA + Llama + MobileViT
"""

from ai_image_detector import AIImageDetector
from mobilevit_detector import MobileViTDetector
from config import LLAVA_API_CONFIG, LLAMA_API_CONFIG
import os


class HybridAIDetector:
    """
    Détecteur hybride utilisant plusieurs modèles
    """
    def __init__(self, use_mobilevit=True):
        """
        Args:
            use_mobilevit: Utiliser MobileViT en plus de LLaVA+Llama
        """
        print("🔧 Initialisation du détecteur hybride...")
        
        # Détecteur LLaVA + Llama
        self.llm_detector = AIImageDetector(
            llava_api_key=LLAVA_API_CONFIG["api_key"],
            llava_base_url=LLAVA_API_CONFIG["base_url"],
            llama_api_key=LLAMA_API_CONFIG["api_key"],
            llama_base_url=LLAMA_API_CONFIG["base_url"]
        )
        
        # Détecteur MobileViT
        self.use_mobilevit = use_mobilevit
        self.mobilevit_detector = None
        
        if use_mobilevit:
            model_path = 'models/mobilevit_best.pth'
            if os.path.exists(model_path):
                self.mobilevit_detector = MobileViTDetector(model_path)
                print("✅ MobileViT chargé")
            else:
                print("⚠️  MobileViT non disponible (modèle non entraîné)")
                print("   Utilisation de LLaVA + Llama uniquement")
                self.use_mobilevit = False
        
        print("✅ Détecteur hybride initialisé\n")
    
    def detect(self, image_path, save_results=True):
        """
        Détecte si une image est générée par IA en combinant plusieurs modèles
        
        Args:
            image_path: Chemin vers l'image
            save_results: Sauvegarder les résultats
            
        Returns:
            Résultat combiné
        """
        print(f"🔍 Analyse hybride de : {image_path}\n")
        
        # 1. Analyse LLaVA + Llama
        print("📝 Étape 1/2 : Analyse LLaVA + Llama...")
        llm_result = self.llm_detector.detect(image_path, save_results=False)
        
        if not llm_result.get("success"):
            return llm_result
        
        llm_score = llm_result["score_global"]
        print(f"   Score LLM : {llm_score:.2f}")
        
        # 2. Analyse MobileViT (si disponible)
        mobilevit_score = None
        if self.use_mobilevit and self.mobilevit_detector:
            print("\n🧠 Étape 2/2 : Analyse MobileViT...")
            mobilevit_result = self.mobilevit_detector.predict(image_path)
            mobilevit_score = mobilevit_result["ai_probability"]
            print(f"   Score MobileViT : {mobilevit_score:.2f}")
        
        # 3. Combiner les scores
        if mobilevit_score is not None:
            # Moyenne pondérée : LLM 60%, MobileViT 40%
            final_score = (llm_score * 0.6) + (mobilevit_score * 0.4)
            method = "Hybride (LLM + MobileViT)"
        else:
            final_score = llm_score
            method = "LLM uniquement"
        
        # 4. Déterminer la conclusion
        if final_score < 0.3:
            conclusion = "Très probablement réelle"
        elif final_score < 0.5:
            conclusion = "Probablement réelle"
        elif final_score < 0.7:
            conclusion = "Incertain"
        elif final_score < 0.85:
            conclusion = "Probablement générée par IA"
        else:
            conclusion = "Très probablement générée par IA"
        
        # 5. Créer le résultat final
        result = {
            "success": True,
            "image_path": image_path,
            "method": method,
            "scores": {
                "llm_score": llm_score,
                "mobilevit_score": mobilevit_score,
                "final_score": final_score
            },
            "conclusion": conclusion,
            "llm_details": {
                "description": llm_result.get("description"),
                "scores_criteres": llm_result.get("scores_criteres"),
                "indices_majeurs": llm_result.get("indices_majeurs"),
                "explication_detaillee": llm_result.get("explication_detaillee")
            }
        }
        
        # Afficher le résultat
        print(f"\n{'=' * 80}")
        print("📊 RÉSULTAT FINAL")
        print('=' * 80)
        print(f"\n🎯 Méthode : {method}")
        print(f"📈 Score final : {final_score:.2f}")
        print(f"💡 Conclusion : {conclusion}")
        
        if mobilevit_score is not None:
            print(f"\n📊 Détails des scores :")
            print(f"   - LLM (60%) : {llm_score:.2f}")
            print(f"   - MobileViT (40%) : {mobilevit_score:.2f}")
        
        print('=' * 80)
        
        # Sauvegarder si demandé
        if save_results:
            import json
            from pathlib import Path
            output_file = f"hybrid_result_{Path(image_path).stem}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n💾 Résultat sauvegardé : {output_file}")
        
        return result


# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python hybrid_detector.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Créer le détecteur hybride
    detector = HybridAIDetector(use_mobilevit=True)
    
    # Analyser l'image
    result = detector.detect(image_path)
