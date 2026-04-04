"""
Pipeline complet de détection d'images générées par IA
Combine LLaVA pour la description et Llama pour le scoring
"""

from image_description_generator import ImageDescriptionGenerator
from ai_image_scorer import AIImageScorer
from config import LLAVA_API_CONFIG, LLAMA_API_CONFIG
from typing import Dict, Any
import json
from pathlib import Path


class AIImageDetector:
    """
    Détecteur complet d'images générées par IA
    """
    
    def __init__(
        self,
        llava_api_key: str,
        llava_base_url: str,
        llama_api_key: str,
        llama_base_url: str,
        llava_model: str = "hosted_vllm/llava-1.5-7b-hf",
        llama_model: str = "hosted_vllm/Llama-3.1-70B-Instruct"
    ):
        """
        Initialise le détecteur avec les deux modèles
        
        Args:
            llava_api_key: Clé API pour LLaVA
            llava_base_url: URL de base pour LLaVA
            llama_api_key: Clé API pour Llama
            llama_base_url: URL de base pour Llama
            llava_model: Modèle LLaVA à utiliser
            llama_model: Modèle Llama à utiliser
        """
        # Initialiser le générateur de description
        self.description_generator = ImageDescriptionGenerator(
            api_key=llava_api_key,
            base_url=llava_base_url,
            model=llava_model
        )
        
        # Initialiser le scorer
        self.scorer = AIImageScorer(
            api_key=llama_api_key,
            base_url=llama_base_url,
            model=llama_model
        )
    
    def detect(self, image_path: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Détecte si une image est générée par IA
        
        Args:
            image_path: Chemin vers l'image à analyser
            save_results: Si True, sauvegarde les résultats dans un fichier JSON
            
        Returns:
            Dictionnaire complet avec description, scores et conclusion
        """
        print(f"🔍 Analyse de l'image : {image_path}\n")
        
        # Étape 1 : Générer la description avec LLaVA
        print("📝 Étape 1/2 : Génération de la description détaillée...")
        description_result = self.description_generator.generate_description(image_path)
        
        if not description_result.get("success"):
            return {
                "success": False,
                "error": f"Erreur lors de la génération de description : {description_result.get('error')}",
                "image_path": image_path
            }
        
        print("✅ Description générée\n")
        
        # Étape 2 : Scorer la description avec Llama
        print("🎯 Étape 2/2 : Analyse et scoring de la description...")
        scoring_result = self.scorer.score_image_description(
            description_result["description"]
        )
        
        if not scoring_result.get("success"):
            return {
                "success": False,
                "error": f"Erreur lors du scoring : {scoring_result.get('error')}",
                "image_path": image_path,
                "description": description_result["description"]
            }
        
        print("✅ Scoring terminé\n")
        
        # Combiner les résultats
        final_result = {
            "success": True,
            "image_path": image_path,
            "description": description_result["description"],
            "scores_criteres": scoring_result.get("scores_criteres", {}),
            "score_global": scoring_result.get("score_global", 0.0),
            "score_global_calcule": scoring_result.get("score_global_calcule", 0.0),
            "conclusion": scoring_result.get("conclusion", "Incertain"),
            "explication_detaillee": scoring_result.get("explication_detaillee", ""),
            "indices_majeurs": scoring_result.get("indices_majeurs", []),
            "niveau_confiance": scoring_result.get("niveau_confiance", "Moyen"),
            "models_used": {
                "description": description_result.get("model"),
                "scoring": scoring_result.get("model")
            }
        }
        
        # Sauvegarder les résultats si demandé
        if save_results:
            output_file = self._get_output_filename(image_path)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(final_result, f, ensure_ascii=False, indent=2)
            final_result["output_file"] = output_file
            print(f"💾 Résultats sauvegardés dans : {output_file}\n")
        
        return final_result
    
    def _get_output_filename(self, image_path: str) -> str:
        """
        Génère un nom de fichier pour les résultats
        
        Args:
            image_path: Chemin de l'image
            
        Returns:
            Nom du fichier de sortie
        """
        image_name = Path(image_path).stem
        return f"detection_result_{image_name}.json"
    
    def print_results(self, result: Dict[str, Any]):
        """
        Affiche les résultats de manière formatée
        
        Args:
            result: Résultat de la détection
        """
        if not result.get("success"):
            print(f"❌ Erreur : {result.get('error')}")
            return
        
        print("=" * 80)
        print("📊 RÉSULTATS DE LA DÉTECTION")
        print("=" * 80)
        print(f"\n🖼️  Image : {result['image_path']}")
        print(f"\n🎯 Conclusion : {result['conclusion']}")
        print(f"📈 Score global : {result['score_global']:.2f} / 1.0")
        print(f"🔍 Niveau de confiance : {result['niveau_confiance']}")
        
        print("\n📋 Scores par critère :")
        print("-" * 80)
        for criterion, score in result['scores_criteres'].items():
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {criterion:35s} : {bar} {score:.2f}")
        
        print("\n💡 Indices majeurs détectés :")
        for i, indice in enumerate(result.get('indices_majeurs', []), 1):
            print(f"  {i}. {indice}")
        
        print(f"\n📝 Explication détaillée :")
        print("-" * 80)
        print(result['explication_detaillee'])
        print("=" * 80)


# Exemple d'utilisation
if __name__ == "__main__":
    # Créer le détecteur
    detector = AIImageDetector(
        llava_api_key=LLAVA_API_CONFIG["api_key"],
        llava_base_url=LLAVA_API_CONFIG["base_url"],
        llama_api_key=LLAMA_API_CONFIG["api_key"],
        llama_base_url=LLAMA_API_CONFIG["base_url"],
        llava_model=LLAVA_API_CONFIG["model"],
        llama_model=LLAMA_API_CONFIG["model"]
    )
    
    # Analyser une image
    image_path = "test_image.jpg"  # Remplace par ton image
    
    result = detector.detect(image_path, save_results=True)
    
    # Afficher les résultats
    detector.print_results(result)
