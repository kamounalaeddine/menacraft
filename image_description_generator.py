"""
Générateur de description détaillée d'images pour la détection d'images IA
Utilise LLaVA-1.5-7B pour analyser tous les critères pertinents
"""

import base64
import httpx
from openai import OpenAI
from typing import Dict, Any
from pathlib import Path


class ImageDescriptionGenerator:
    """
    Génère une description technique et détaillée d'une image
    en analysant tous les critères de détection d'images IA
    """
    
    def __init__(self, api_key: str, base_url: str, model: str = "hosted_vllm/llava-1.5-7b-hf"):
        """
        Initialise le générateur avec l'API OpenAI-compatible
        
        Args:
            api_key: Clé API
            base_url: URL de base de l'API
            model: Nom du modèle LLaVA à utiliser
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
    
    def encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode une image en base64 pour l'envoyer à l'API
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Image encodée en base64
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def get_image_data_url(self, image_path: str) -> str:
        """
        Crée une data URL pour l'image
        
        Args:
            image_path: Chemin vers l'image
            
        Returns:
            Data URL de l'image
        """
        # Déterminer le type MIME
        extension = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        mime_type = mime_types.get(extension, 'image/jpeg')
        
        # Encoder l'image
        image_base64 = self.encode_image_to_base64(image_path)
        
        return f"data:{mime_type};base64,{image_base64}"
    
    def create_detailed_prompt(self) -> str:
        """
        Crée le prompt détaillé pour l'analyse de l'image
        
        Returns:
            Prompt structuré avec tous les critères
        """
        prompt = """Tu es un expert TRÈS CRITIQUE en détection d'images générées par IA. Ta mission est de CHERCHER ACTIVEMENT les anomalies et les indices de génération IA.

IMPORTANT : Sois ATTENTIF aux détails suspects, mais reste OBJECTIF. Ne force pas les conclusions.

Analyse cette image de manière TRÈS DÉTAILLÉE et CRITIQUE selon les critères suivants :

**1. TEXTURES ET PATTERNS** ⚠️ CHERCHE LES ANOMALIES
- Y a-t-il des répétitions EXACTEMENT identiques (copier-coller) ?
- Les textures sont-elles TROP uniformes ou TROP parfaites ?
- Y a-t-il des patterns géométriques qui se répètent de manière suspecte ?
- Les textures ont-elles un aspect "synthétique" ou "plastique" ?
- Y a-t-il des watermarks IA subtils ou des artefacts de génération ?

**2. CONTOURS ET BORDS**
- Les contours des objets sont-ils nets ou flous de manière suspecte ?
- Y a-t-il des transitions abruptes ou des "bleeding" de couleurs ?
- Les bords sont-ils cohérents ou présentent des anomalies (flou sélectif bizarre) ?
- Y a-t-il des contours qui "fondent" ou se mélangent anormalement ?

**3. PROPORTIONS ET PERSPECTIVE**
- Les proportions des objets sont-elles correctes (taille, échelle) ?
- La perspective est-elle cohérente dans toute l'image ?
- Y a-t-il des distorsions ou des incohérences spatiales ?
- Les objets en arrière-plan respectent-ils les lois de la perspective ?

**4. COULEURS ET GRADIENTS**
- Les couleurs sont-elles naturelles ou saturées artificiellement ?
- Les gradients sont-ils lisses ou présentent des bandes (posterization) ?
- Y a-t-il des transitions de couleur suspectes ou des halos ?
- La palette de couleurs est-elle cohérente ou trop "parfaite" ?

**5. ANATOMIE ET DÉTAILS HUMAINS (si présents)** ⚠️ IMPORTANT
- Les mains : Si visibles, observe attentivement le nombre de doigts et leur position
  * Décris ce que tu vois sans préjugé
  * Les articulations semblent-elles naturelles ?
  * Les ongles sont-ils présents ?
- Les visages : Observe les proportions et la symétrie
- Les yeux : Vérifie la cohérence des pupilles et des reflets
- Les cheveux : Note le niveau de détail
- La peau : Décris la texture observée

⚠️ IMPORTANT : Décris OBJECTIVEMENT ce que tu vois, sans chercher à forcer une conclusion.

**6. TEXTE ET ÉCRITURE (si présents)** ⚠️ CRITIQUE
- Le texte est-il PARFAITEMENT lisible mot par mot ?
- Y a-t-il des lettres déformées, floues, ou qui ressemblent à des lettres mais n'en sont pas ?
- Les mots ont-ils un sens OU sont-ce des suites de caractères aléatoires ?
- Les panneaux, logos, enseignes sont-ils EXACTEMENT corrects ?
- SIGNALE tout texte illisible ou bizarre, c'est un FORT indice d'IA

**7. OBJETS ET DÉTAILS COMPLEXES**
- Les objets sont-ils cohérents et réalistes ?
- Y a-t-il des détails manquants ou incohérents ?
- Les objets complexes (bijoux, montres, lunettes) sont-ils corrects ?
- Les éléments répétitifs (fenêtres, briques) sont-ils cohérents ?

**8. SYMÉTRIES ET PATTERNS**
- Y a-t-il des symétries parfaites anormales ?
- Les éléments symétriques sont-ils naturels ou artificiels ?
- Les patterns répétitifs sont-ils trop parfaits ?

**9. BRUIT, GRAIN ET NETTETÉ**
- Quel est le niveau de bruit dans l'image ?
- Le bruit est-il distribué naturellement ou uniformément ?
- Y a-t-il des zones trop lisses (peau plastique) ou trop bruitées ?
- La netteté est-elle cohérente ou sélective de manière suspecte ?

**10. COMPRESSION ET ARTEFACTS**
- Y a-t-il des artefacts de compression visibles ?
- Des blocs ou des distorsions JPEG ?
- Des anomalies de compression suspectes ?
- Présence d'artefacts typiques des GANs (checkerboard, banding) ?

**11. LUMIÈRE, OMBRES ET REFLETS**
- L'éclairage est-il cohérent dans toute l'image ?
- Les ombres sont-elles réalistes, bien placées et de la bonne intensité ?
- Y a-t-il des incohérences dans la direction de la lumière ?
- Les reflets sur surfaces brillantes sont-ils corrects ?
- Les ombres portées correspondent-elles aux objets ?

**12. PROFONDEUR DE CHAMP ET BOKEH**
- Le flou d'arrière-plan (bokeh) est-il naturel ?
- La profondeur de champ est-elle cohérente avec l'optique ?
- Y a-t-il des zones floues de manière incohérente ?

**13. MÉTADONNÉES ET INDICES TECHNIQUES**
- L'image semble-t-elle avoir été post-traitée ?
- Y a-t-il des indices de génération IA (artefacts spécifiques) ?
- La résolution et la qualité sont-elles cohérentes ?

**14. COHÉRENCE GLOBALE ET CONTEXTE**
- L'image est-elle globalement cohérente ?
- Y a-t-il des éléments qui semblent "collés" ou artificiels ?
- Le style est-il uniforme dans toute l'image ?
- Le contexte et la scène sont-ils logiques et réalistes ?
- Y a-t-il des impossibilités physiques ou des incohérences narratives ?

Pour chaque critère, fournis :
- Une description CRITIQUE de ce que tu observes
- Des exemples PRÉCIS d'anomalies (même mineures)
- Ton impression : NATUREL, SUSPECT, ou ARTIFICIEL

⚠️ RÈGLES IMPORTANTES : 
1. Décris OBJECTIVEMENT ce que tu vois
2. Signale les anomalies RÉELLES, pas imaginées
3. Les images IA modernes sont très réalistes, cherche les VRAIS détails suspects
4. Concentre-toi sur : MAINS (si visibles clairement), TEXTE, OMBRES, ARRIÈRE-PLAN

Sois PRÉCIS, TECHNIQUE et OBJECTIF dans tes observations. CHERCHE les problèmes réels."""

        return prompt

    
    def generate_description(self, image_path: str) -> Dict[str, Any]:
        """
        Génère une description détaillée de l'image
        
        Args:
            image_path: Chemin vers l'image à analyser
            
        Returns:
            Dictionnaire contenant la description détaillée et les métadonnées
        """
        # Vérifier que l'image existe
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image non trouvée : {image_path}")
        
        # Créer la data URL de l'image
        image_data_url = self.get_image_data_url(image_path)
        
        # Créer le prompt
        prompt = self.create_detailed_prompt()
        
        try:
            # Appeler l'API avec le client OpenAI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1200,  # Réduit pour éviter de dépasser la limite du contexte
                temperature=0.3
            )
            
            # Extraire la description
            description = response.choices[0].message.content
            
            return {
                "success": True,
                "description": description,
                "image_path": image_path,
                "model": self.model
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }
    
    def generate_description_with_custom_criteria(
        self, 
        image_path: str, 
        custom_criteria: list[str]
    ) -> Dict[str, Any]:
        """
        Génère une description avec des critères personnalisés
        
        Args:
            image_path: Chemin vers l'image
            custom_criteria: Liste de critères personnalisés à analyser
            
        Returns:
            Dictionnaire avec la description
        """
        # Créer la data URL de l'image
        image_data_url = self.get_image_data_url(image_path)
        
        # Créer un prompt personnalisé
        criteria_text = "\n".join([f"- {criterion}" for criterion in custom_criteria])
        prompt = f"""Analyse cette image selon les critères suivants :

{criteria_text}

Pour chaque critère, fournis une analyse détaillée et technique."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url}
                            }
                        ]
                    }
                ],
                max_tokens=1200,  # Réduit
                temperature=0.3
            )
            
            description = response.choices[0].message.content
            
            return {
                "success": True,
                "description": description,
                "image_path": image_path,
                "custom_criteria": custom_criteria
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "image_path": image_path
            }


# Exemple d'utilisation
if __name__ == "__main__":
    from config import LLAVA_API_CONFIG
    
    # Créer le générateur avec la configuration
    generator = ImageDescriptionGenerator(
        api_key=LLAVA_API_CONFIG["api_key"],
        base_url=LLAVA_API_CONFIG["base_url"],
        model=LLAVA_API_CONFIG["model"]
    )
    
    # Générer la description
    image_path = "test_image.jpg"  # Remplace par le chemin de ton image
    
    print("🔍 Génération de la description détaillée...")
    result = generator.generate_description(image_path)
    
    if result["success"]:
        print("\n✅ Description générée avec succès !\n")
        print("=" * 80)
        print(result["description"])
        print("=" * 80)
    else:
        print(f"\n❌ Erreur : {result['error']}")
