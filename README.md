# Agent de Validation de Logique d'Image

Cet agent extrait le texte d'une image et vérifie si son contenu est logique et cohérent.

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### Utilisation basique

```python
from image_logic_validator import ImageLogicValidator

# Créer l'agent
agent = ImageLogicValidator(api_key="sk-f90b3b0c49044060b61a6dd33aa6b827")

# Analyser une image
resultat = agent.validate_image_text("mon_image.jpg")

# Afficher les résultats
print(f"Texte: {resultat['texte_extrait']}")
print(f"Logique: {resultat['est_logique']}")
print(f"Explication: {resultat['explication']}")
```

### Exécution directe

```bash
python image_logic_validator.py
```

N'oublie pas de modifier `IMAGE_PATH` dans le code avec le chemin de ton image.

## Fonctionnalités

- ✅ Extraction de texte depuis une image (OCR via LLaVA)
- ✅ Vérification de la cohérence logique
- ✅ Analyse de la validité du contenu
- ✅ Détection des contradictions
- ✅ Vérification grammaticale

## Configuration

Modifie les paramètres dans le code si nécessaire :
- `API_KEY` : Ta clé API
- `model` : Le modèle LLaVA à utiliser
- `temperature` : Créativité du modèle (0.0 à 1.0)
- `max_tokens` : Longueur maximale de la réponse
