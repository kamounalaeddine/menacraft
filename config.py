"""
Configuration pour le système de détection d'images IA
"""

# Configuration de l'API LLaVA
LLAVA_API_CONFIG = {
    "api_key": "sk-f90b3b0c49044060b61a6dd33aa6b827",
    "base_url": "https://tokenfactory.esprit.tn/api",
    "model": "hosted_vllm/llava-1.5-7b-hf",
    "timeout": 60,
    "max_tokens": 2000,
    "temperature": 0.3
}

# Configuration de l'API Llama pour le scoring
LLAMA_API_CONFIG = {
    "api_key": "sk-f90b3b0c49044060b61a6dd33aa6b827",
    "base_url": "https://tokenfactory.esprit.tn/api",
    "model": "hosted_vllm/Llama-3.1-70B-Instruct",
    "timeout": 60,
    "max_tokens": 1500,
    "temperature": 0.7
}

# Critères de détection d'images IA
DETECTION_CRITERIA = [
    "textures_patterns",
    "contours_bords",
    "proportions_perspective",
    "couleurs_gradients",
    "anatomie_details_humains",
    "texte_ecriture",
    "objets_details_complexes",
    "symetries_patterns",
    "bruit_grain_nettete",
    "compression_artefacts",
    "lumiere_ombres_reflets",
    "profondeur_champ_bokeh",
    "metadonnees_indices_techniques",
    "coherence_globale_contexte"
]

# Pondération des critères pour le score final
CRITERIA_WEIGHTS = {
    "textures_patterns": 0.09,
    "contours_bords": 0.06,
    "proportions_perspective": 0.08,
    "couleurs_gradients": 0.07,
    "anatomie_details_humains": 0.20,  # TRÈS important (mains, visages) - AUGMENTÉ
    "texte_ecriture": 0.15,  # TRÈS important (texte souvent raté par IA) - AUGMENTÉ
    "objets_details_complexes": 0.08,
    "symetries_patterns": 0.06,
    "bruit_grain_nettete": 0.06,
    "compression_artefacts": 0.04,
    "lumiere_ombres_reflets": 0.08,  # Important (physique de la lumière)
    "profondeur_champ_bokeh": 0.02,
    "metadonnees_indices_techniques": 0.01,
    "coherence_globale_contexte": 0.00  # Moins important car trop subjectif
}
