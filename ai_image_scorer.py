"""
Module de scoring pour déterminer si une image est générée par IA
Utilise Llama-3.1-70B-Instruct pour analyser la description et donner des scores
"""

import httpx
from openai import OpenAI
from typing import Dict, Any
import json
from config import LLAMA_API_CONFIG, DETECTION_CRITERIA, CRITERIA_WEIGHTS


class AIImageScorer:
    """
    Analyse une description d'image et détermine la probabilité
    qu'elle soit générée par IA
    """
    
    def __init__(self, api_key: str, base_url: str, model: str = "hosted_vllm/Llama-3.1-70B-Instruct"):
        """
        Initialise le scorer avec l'API Llama
        
        Args:
            api_key: Clé API
            base_url: URL de base de l'API
            model: Nom du modèle Llama à utiliser
        """
        self.model = model
        
        # Créer un client HTTP qui désactive la vérification SSL
        http_client = httpx.Client(verify=False)
        
        # Initialiser le client OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client
        )
    
    def create_scoring_prompt(self, image_description: str) -> str:
        """
        Crée le prompt de scoring pour Llama
        
        Args:
            image_description: Description détaillée de l'image générée par LLaVA
            
        Returns:
            Prompt structuré pour le scoring
        """
        prompt = f"""Tu es un expert TRÈS STRICT en détection d'images générées par intelligence artificielle.

Une description détaillée de l'image a été fournie ci-dessous. Ta mission est d'analyser cette description et de déterminer la probabilité que l'image soit générée par IA.

⚠️ IMPORTANT : Sois CRITIQUE et SENSIBLE aux anomalies. Même les petits indices doivent augmenter le score.

**DESCRIPTION DE L'IMAGE :**
{image_description}

---

**INSTRUCTIONS :**

Pour chaque critère ci-dessous, évalue la probabilité que l'image soit générée par IA en donnant un **score entre 0 et 1** :
- **0.0** = Aucun signe d'IA (parfaitement naturel)
- **0.3** = Indices mineurs suspects
- **0.5** = Indices modérés (doute raisonnable)
- **0.7** = Indices forts d'IA
- **1.0** = Très probablement IA (signes évidents)

⚠️ RÈGLES DE SCORING STRICTES :
- Toute anomalie anatomique (doigts, yeux, etc.) = MINIMUM 0.9
- Texte illisible ou bizarre = MINIMUM 0.9
- Textures trop parfaites ou répétitives = MINIMUM 0.7
- Ombres incohérentes = MINIMUM 0.8
- Arrière-plan flou suspect = MINIMUM 0.6
- Objets qui se mélangent = MINIMUM 0.7
- En cas de doute, AUGMENTE le score plutôt que de le diminuer

**EXEMPLES DE SCORING :**

Exemple 1 - Main avec 6 doigts :
- anatomie_details_humains: 1.0 (erreur majeure)
- score_global: ≥ 0.8

Exemple 2 - Texte flou/illisible :
- texte_ecriture: 1.0 (fort indice IA)
- score_global: ≥ 0.8

Exemple 3 - Ombres dans mauvaise direction :
- lumiere_ombres_reflets: 0.9 (physique incorrecte)
- score_global: ≥ 0.7

Exemple 4 - Textures répétitives :
- textures_patterns: 0.8 (pattern suspect)
- score_global: ≥ 0.6

Exemple 5 - Arrière-plan qui fond :
- contours_bords: 0.7
- coherence_globale_contexte: 0.8
- score_global: ≥ 0.6

**CRITÈRES À ÉVALUER :**

1. **textures_patterns** : Répétitions anormales, motifs trop uniformes, watermarks IA
2. **contours_bords** : Contours flous suspects, bleeding de couleurs, transitions anormales
3. **proportions_perspective** : Incohérences spatiales, distorsions, perspective incorrecte
4. **couleurs_gradients** : Saturation artificielle, bandes, halos, palette trop parfaite
5. **anatomie_details_humains** : Mains (doigts), visages, yeux, cheveux, peau - anomalies anatomiques
6. **texte_ecriture** : Texte illisible, lettres déformées, caractères inventés
7. **objets_details_complexes** : Objets incohérents, détails manquants, éléments impossibles
8. **symetries_patterns** : Symétries parfaites anormales, patterns trop réguliers
9. **bruit_grain_nettete** : Distribution de bruit suspecte, zones trop lisses, netteté incohérente
10. **compression_artefacts** : Artefacts GAN (checkerboard, banding), compression anormale
11. **lumiere_ombres_reflets** : Éclairage incohérent, ombres irréalistes, reflets incorrects
12. **profondeur_champ_bokeh** : Bokeh artificiel, flou incohérent, profondeur de champ impossible
13. **metadonnees_indices_techniques** : Indices de post-traitement IA, artefacts spécifiques
14. **coherence_globale_contexte** : Éléments collés, incohérences narratives, impossibilités physiques

**INDICES FORTS D'IMAGE IA (Score élevé obligatoire) :**
- 🖐️ Mains avec nombre de doigts ≠ 5, articulations impossibles → SCORE ≥ 0.9
- 📝 Texte illisible, lettres inventées, mots sans sens → SCORE ≥ 0.9
- 👁️ Asymétries faciales majeures, dents bizarres → SCORE ≥ 0.8
- 💡 Ombres incohérentes, éclairage impossible → SCORE ≥ 0.8
- 🔄 Symétries parfaites non naturelles, patterns répétitifs → SCORE ≥ 0.7
- 🌫️ Arrière-plans qui fondent, détails qui disparaissent → SCORE ≥ 0.7
- 🎨 Textures trop uniformes ou répétitives → SCORE ≥ 0.7
- 🔍 Objets qui se mélangent anormalement → SCORE ≥ 0.7

🚨 **RED FLAGS ABSOLUS (Score = 1.0 immédiat) :**
- Main avec 4, 6, 7+ doigts
- Doigts fusionnés ou qui se transforment en autre chose
- Texte complètement illisible ou lettres inventées
- Yeux de couleurs différentes sans raison médicale
- Ombres qui pointent dans plusieurs directions
- Objets qui disparaissent progressivement
- Reflets qui ne correspondent à rien

⚠️ SI UN SEUL de ces RED FLAGS est présent, le score global DOIT être ≥ 0.8

**FORMAT DE SORTIE REQUIS :**

Tu DOIS répondre UNIQUEMENT avec un objet JSON valide, sans texte avant ou après. Format exact :

```json
{{
  "scores_criteres": {{
    "textures_patterns": 0.0,
    "contours_bords": 0.0,
    "proportions_perspective": 0.0,
    "couleurs_gradients": 0.0,
    "anatomie_details_humains": 0.0,
    "texte_ecriture": 0.0,
    "objets_details_complexes": 0.0,
    "symetries_patterns": 0.0,
    "bruit_grain_nettete": 0.0,
    "compression_artefacts": 0.0,
    "lumiere_ombres_reflets": 0.0,
    "profondeur_champ_bokeh": 0.0,
    "metadonnees_indices_techniques": 0.0,
    "coherence_globale_contexte": 0.0
  }},
  "score_global": 0.0,
  "conclusion": "Très probablement réelle / Probablement réelle / Incertain / Probablement générée par IA / Très probablement générée par IA",
  "explication_detaillee": "Explication détaillée des anomalies ou indices principaux qui justifient la conclusion",
  "indices_majeurs": ["Liste des 3-5 indices les plus importants détectés"],
  "niveau_confiance": "Faible / Moyen / Élevé / Très élevé"
}}
```

**IMPORTANT :** 
- Réponds UNIQUEMENT avec le JSON, rien d'autre
- Tous les scores doivent être entre 0.0 et 1.0
- Le score_global doit refléter la présence d'anomalies
- Sois STRICT : en cas de doute, AUGMENTE le score
- Un seul indice fort suffit pour un score global ≥ 0.6
- Sois précis et factuel dans ton explication"""

        return prompt

    
    def calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """
        Calcule le score global pondéré avec boost pour les anomalies critiques
        
        Args:
            scores: Dictionnaire des scores par critère
            
        Returns:
            Score global pondéré entre 0 et 1
        """
        total_score = 0.0
        total_weight = 0.0
        
        # Vérifier les critères critiques
        critical_criteria = {
            "anatomie_details_humains": 0.9,  # Si score > 0.5, boost
            "texte_ecriture": 0.9,
            "lumiere_ombres_reflets": 0.8
        }
        
        has_critical_issue = False
        max_critical_score = 0.0
        
        for criterion, score in scores.items():
            if criterion in CRITERIA_WEIGHTS:
                weight = CRITERIA_WEIGHTS[criterion]
                total_score += score * weight
                total_weight += weight
                
                # Vérifier si c'est un critère critique avec score élevé
                if criterion in critical_criteria and score >= 0.5:
                    has_critical_issue = True
                    max_critical_score = max(max_critical_score, score)
        
        base_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Si anomalie critique détectée, augmenter le score global
        if has_critical_issue:
            # Boost le score en fonction de la sévérité de l'anomalie critique
            boost = (max_critical_score - 0.6) * 0.25  # Boost réduit, seuil augmenté
            final_score = min(1.0, base_score + boost)
            return final_score
        
        return base_score
    
    def score_image_description(self, image_description: str) -> Dict[str, Any]:
        """
        Score une description d'image pour déterminer si elle est générée par IA
        
        Args:
            image_description: Description détaillée de l'image
            
        Returns:
            Dictionnaire avec scores, conclusion et explication
        """
        # Créer le prompt
        prompt = self.create_scoring_prompt(image_description)
        
        try:
            # Appeler l'API Llama
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert EXTRÊMEMENT STRICT en détection d'images IA. Tu réponds UNIQUEMENT avec du JSON valide. Tu es HYPER-CRITIQUE et TRÈS SENSIBLE aux anomalies. Même les petites anomalies doivent avoir un score élevé."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.25,  # Augmenté pour plus de flexibilité
                max_tokens=1500,
                top_p=0.92  # Augmenté pour plus de variété
            )
            
            # Extraire la réponse
            response_text = response.choices[0].message.content.strip()
            
            # Nettoyer la réponse (enlever les markdown code blocks si présents)
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            # Parser le JSON
            result = json.loads(response_text)
            
            # Vérifier et recalculer le score global si nécessaire
            if "scores_criteres" in result:
                calculated_score = self.calculate_weighted_score(result["scores_criteres"])
                result["score_global_calcule"] = calculated_score
                
                # Ajouter des métadonnées
                result["success"] = True
                result["model"] = self.model
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Erreur de parsing JSON : {str(e)}",
                "raw_response": response_text if 'response_text' in locals() else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_conclusion_from_score(self, score: float) -> str:
        """
        Détermine la conclusion textuelle basée sur le score
        
        Args:
            score: Score global entre 0 et 1
            
        Returns:
            Conclusion textuelle
        """
        if score < 0.2:
            return "Très probablement réelle"
        elif score < 0.4:
            return "Probablement réelle"
        elif score < 0.6:
            return "Incertain"
        elif score < 0.8:
            return "Probablement générée par IA"
        else:
            return "Très probablement générée par IA"
    
    def get_confidence_level(self, scores: Dict[str, float]) -> str:
        """
        Détermine le niveau de confiance basé sur la variance des scores
        
        Args:
            scores: Dictionnaire des scores par critère
            
        Returns:
            Niveau de confiance
        """
        if not scores:
            return "Faible"
        
        values = list(scores.values())
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        # Faible variance = haute confiance
        if variance < 0.05:
            return "Très élevé"
        elif variance < 0.15:
            return "Élevé"
        elif variance < 0.25:
            return "Moyen"
        else:
            return "Faible"


# Exemple d'utilisation
if __name__ == "__main__":
    from config import LLAMA_API_CONFIG
    
    # Créer le scorer
    scorer = AIImageScorer(
        api_key=LLAMA_API_CONFIG["api_key"],
        base_url=LLAMA_API_CONFIG["base_url"],
        model=LLAMA_API_CONFIG["model"]
    )
    
    # Exemple de description (normalement générée par LLaVA)
    test_description = """
    L'image montre un portrait d'une personne. Les mains présentent 6 doigts sur la main droite,
    ce qui est anatomiquement incorrect. Le texte en arrière-plan est flou et illisible.
    Les ombres ne correspondent pas à la direction de la lumière principale.
    La texture de la peau est trop lisse et uniforme, sans pores visibles.
    """
    
    print("🔍 Analyse de la description...\n")
    result = scorer.score_image_description(test_description)
    
    if result.get("success"):
        print("✅ Scoring réussi !\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"❌ Erreur : {result.get('error')}")
