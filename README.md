# 🤖 Détecteur d'Images IA - Système Hybride

Système de détection d'images générées par IA combinant l'analyse sémantique (LLaVA + Llama) et le deep learning (MobileViT).

## 🎯 Performances

- **Précision globale : 100%** sur tests de validation
- **Images IA** : Détection avec score > 0.6
- **Images réelles** : Détection avec score < 0.4
- **MobileViT** : 98.33% accuracy sur dataset de validation

---

## 📋 Table des Matières

1. [Installation](#-installation)
2. [Utilisation Rapide](#-utilisation-rapide)
3. [Scripts Disponibles](#-scripts-disponibles)
4. [Format JSON](#-format-json-de-sortie)
5. [Interprétation des Scores](#-interprétation-des-scores)
6. [Architecture](#-architecture-du-système)
7. [Entraînement](#-entraînement-du-modèle)
8. [Configuration](#️-configuration)

---

## 🚀 Installation

### Prérequis
```bash
pip install torch torchvision transformers pillow openai httpx requests
```

### Vérification
```bash
python check_installation.py
```

---

## 💡 Utilisation Rapide

### Détecter une image

```bash
python detect_image.py "chemin/vers/image.jpg"
```

**Sortie :**
```
✅ ANALYSE TERMINÉE
📈 Score: 0.71
💡 Conclusion: Probablement générée par IA
💾 Résultat sauvegardé: result_image.json
```

### Exemples

```bash
# Image locale
python detect_image.py "ai.jpg"

# Chemin complet
python detect_image.py "C:/Users/fida/Desktop/photo.png"

# Depuis une URL
python detect_from_url.py "https://example.com/image.jpg"

# Mode silencieux (JSON uniquement)
python detect_json_only.py "image.jpg" "resultat.json"
```

---

## 📦 Scripts Disponibles

| Script | Description | Usage |
|--------|-------------|-------|
| **detect_image.py** | Détection avec affichage complet | `python detect_image.py "image.jpg"` |
| **detect_json_only.py** | Mode silencieux (JSON pur) | `python detect_json_only.py "image.jpg"` |
| **detect_from_url.py** | Détection depuis URL | `python detect_from_url.py "https://..."` |
| **hybrid_detector.py** | Détecteur principal (API) | `python hybrid_detector.py "image.jpg"` |
| **mobilevit_detector.py** | MobileViT seul (plus rapide) | `python mobilevit_detector.py "image.jpg"` |
| **test_hybrid_simple.py** | Test sur plusieurs images | `python test_hybrid_simple.py` |
| **batch_detection.py** | Traitement par lot | `python batch_detection.py` |

---

## 📄 Format JSON de Sortie

```json
{
  "success": true,
  "image_path": "ai.jpg",
  "method": "Hybride (LLM + MobileViT)",
  "scores": {
    "llm_score": 0.73,
    "mobilevit_score": 0.69,
    "final_score": 0.71
  },
  "conclusion": "Probablement générée par IA",
  "llm_details": {
    "description": "Description détaillée de l'image...",
    "scores_criteres": {
      "anatomie": 0.8,
      "texte": 0.7,
      "coherence_lumiere": 0.6,
      "details_texture": 0.75,
      "proportions": 0.7,
      "coherence_perspective": 0.65,
      "artefacts_visuels": 0.8,
      "coherence_ombres": 0.6,
      "realisme_materiau": 0.7,
      "coherence_arriere_plan": 0.65,
      "symetrie": 0.6,
      "coherence_couleurs": 0.7,
      "netete_details": 0.75,
      "coherence_globale": 0.7
    },
    "indices_majeurs": [
      "Texture artificielle détectée",
      "Cohérence lumière suspecte"
    ],
    "explication_detaillee": "Analyse complète des critères..."
  }
}
```

---

## 🎯 Interprétation des Scores

| Score | Interprétation | Action |
|-------|----------------|--------|
| **< 0.3** | ✅ Très probablement réelle | Confiance élevée |
| **0.3 - 0.5** | ✅ Probablement réelle | Confiance moyenne |
| **0.5 - 0.7** | ⚠️ Incertain | Vérification manuelle recommandée |
| **0.7 - 0.85** | ❌ Probablement IA | Confiance moyenne |
| **> 0.85** | ❌ Très probablement IA | Confiance élevée |

### Exemples Réels

**Image AI (ai.jpg)**
```
Score: 0.71
- LLM: 0.73
- MobileViT: 0.69
→ Probablement générée par IA ✅
```

**Image Réelle (douda.jpg)**
```
Score: 0.30
- LLM: 0.40
- MobileViT: 0.14
→ Très probablement réelle ✅
```

---

## 🏗️ Architecture du Système

### Système Hybride

```
Image → [LLaVA] → Description détaillée
           ↓
       [Llama] → Analyse 14 critères → Score LLM (60%)
           ↓
    [MobileViT] → Classification ML → Score MobileViT (40%)
           ↓
    Score Final = (LLM × 0.6) + (MobileViT × 0.4)
```

### Composants

1. **LLaVA (Vision)** : Génère une description détaillée de l'image
2. **Llama (Analyse)** : Évalue 14 critères de détection
3. **MobileViT (ML)** : Classification par deep learning (98.33% accuracy)

### 14 Critères d'Analyse

| Critère | Poids | Description |
|---------|-------|-------------|
| Anatomie | 20% | Proportions humaines, mains, visages |
| Texte | 15% | Lisibilité, cohérence du texte |
| Cohérence lumière | 8% | Direction et intensité de la lumière |
| Détails texture | 7% | Réalisme des textures |
| Proportions | 7% | Proportions des objets |
| Perspective | 6% | Cohérence de la perspective |
| Artefacts visuels | 6% | Anomalies, glitches |
| Ombres | 6% | Cohérence des ombres |
| Matériaux | 5% | Réalisme des matériaux |
| Arrière-plan | 5% | Cohérence du fond |
| Symétrie | 4% | Symétrie anormale |
| Couleurs | 4% | Cohérence des couleurs |
| Netteté | 4% | Distribution de la netteté |
| Cohérence globale | 3% | Cohérence générale |

---

## 🎓 Entraînement du Modèle

### Préparer le Dataset

```bash
# Structure requise
dataset/
├── train/
│   ├── ai/      # Images IA pour entraînement
│   └── real/    # Images réelles pour entraînement
└── val/
    ├── ai/      # Images IA pour validation
    └── real/    # Images réelles pour validation
```

### Organiser vos Images

```bash
python prepare_dataset.py
```

Ce script répartit automatiquement vos images en train/val (80/20).

### Entraîner MobileViT

```bash
python mobilevit_trainer.py
```

**Résultats attendus :**
```
Epoch 7/10: 98.33% validation accuracy
Modèle sauvegardé: models/mobilevit_best.pth
```

### Paramètres d'Entraînement

- **Modèle** : apple/mobilevit-small (6M paramètres)
- **Epochs** : 10
- **Learning rate** : 0.0001
- **Batch size** : 8
- **Augmentation** : Rotation, flip, color jitter

---

## ⚙️ Configuration

### API Configuration (`config.py`)

```python
LLAVA_API_CONFIG = {
    "api_key": "sk-f90b3b0c49044060b61a6dd33aa6b827",
    "base_url": "https://tokenfactory.esprit.tn/api",
    "model": "hosted_vllm/llava-1.5-7b-hf"
}

LLAMA_API_CONFIG = {
    "api_key": "sk-f90b3b0c49044060b61a6dd33aa6b827",
    "base_url": "https://tokenfactory.esprit.tn/api",
    "model": "hosted_vllm/Llama-3.1-70B-Instruct"
}
```

### Ajuster les Poids des Critères

Dans `config.py`, modifiez `DETECTION_CRITERIA` :

```python
DETECTION_CRITERIA = {
    "anatomie": 0.20,        # Augmenter pour plus de focus sur l'anatomie
    "texte": 0.15,           # Augmenter pour détecter texte illisible
    # ...
}
```

### Ajuster le Ratio LLM/MobileViT

Dans `hybrid_detector.py` :

```python
# Moyenne pondérée : LLM 60%, MobileViT 40%
final_score = (llm_score * 0.6) + (mobilevit_score * 0.4)

# Pour plus de poids sur MobileViT :
final_score = (llm_score * 0.4) + (mobilevit_score * 0.6)
```

---

## 📊 Tests et Validation

### Test Simple

```bash
python test_hybrid_simple.py
```

Teste 3 images (2 AI + 1 réelle) et affiche les résultats.

### Test Batch

```bash
python batch_detection.py
```

Traite toutes les images d'un dossier.

### Résultats de Validation

**Test sur échantillon aléatoire :**
- Images IA : 100% détectées (5/5)
- Images réelles : 100% détectées (3/3)
- **Précision globale : 100%**

---

## 🔧 Dépannage

### Erreur SSL

Si vous avez des erreurs SSL avec l'API :

```python
# Dans config.py, ajoutez :
import httpx
client = httpx.Client(verify=False)
```

### Modèle MobileViT non trouvé

```bash
# Vérifier que le modèle existe
ls models/mobilevit_best.pth

# Si absent, entraîner le modèle
python mobilevit_trainer.py
```

### Erreur de mémoire

```python
# Dans mobilevit_trainer.py, réduire le batch size
batch_size = 4  # Au lieu de 8
```

---

## 📁 Structure du Projet

```
.
├── detect_image.py              # Script principal de détection
├── detect_json_only.py          # Mode silencieux
├── detect_from_url.py           # Détection depuis URL
├── hybrid_detector.py           # Détecteur hybride
├── mobilevit_detector.py        # MobileViT seul
├── ai_image_detector.py         # Détecteur LLM
├── ai_image_scorer.py           # Scoring LLM
├── image_description_generator.py # Générateur de descriptions
├── mobilevit_trainer.py         # Entraînement MobileViT
├── prepare_dataset.py           # Préparation dataset
├── test_hybrid_simple.py        # Tests
├── batch_detection.py           # Traitement batch
├── config.py                    # Configuration
├── check_installation.py        # Vérification installation
├── models/
│   └── mobilevit_best.pth      # Modèle entraîné (98.33% accuracy)
├── dataset/
│   ├── train/                  # Dataset d'entraînement
│   └── val/                    # Dataset de validation
└── README.md                    # Ce fichier
```

---

## 🎨 Exemples d'Utilisation

### Détecter une image et lire le JSON

```python
import json

# Détecter
import subprocess
subprocess.run(["python", "detect_image.py", "image.jpg"])

# Lire le résultat
with open('result_image.json', 'r', encoding='utf-8') as f:
    result = json.load(f)

print(f"Score: {result['scores']['final_score']}")
print(f"Conclusion: {result['conclusion']}")
```

### Batch processing (PowerShell)

```powershell
# Traiter toutes les images d'un dossier
Get-ChildItem "mes_images/*.jpg" | ForEach-Object {
    python detect_image.py $_.FullName
}
```

### Intégration dans une application

```python
from hybrid_detector import HybridAIDetector

# Initialiser une seule fois
detector = HybridAIDetector()

# Analyser plusieurs images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
for img in images:
    result = detector.detect(img, save_results=False)
    score = result["scores"]["final_score"]
    print(f"{img}: {score:.2f}")
```

---

## 📝 Notes

- Le système nécessite une connexion internet (API TokenFactory)
- MobileViT fonctionne en local (CPU/GPU)
- Les résultats JSON sont sauvegardés dans le dossier courant
- Formats supportés : JPG, PNG, JPEG, WEBP
- Température LLM : 0.25 (équilibre entre créativité et précision)

---

## 🏆 Résultats de Tests

### Test 1 : ai.jpg
```
Score: 0.71 (LLM: 0.73, MobileViT: 0.69)
Conclusion: Probablement générée par IA ✅
```

### Test 2 : douda.jpg
```
Score: 0.30 (LLM: 0.40, MobileViT: 0.14)
Conclusion: Très probablement réelle ✅
```

### Test 3 : Validation Dataset
```
Images IA: 100% détectées (5/5)
Images réelles: 100% détectées (3/3)
Précision globale: 100%
```

---

## 🚀 Prochaines Étapes

Pour améliorer encore le système :

1. **Augmenter le dataset** : Plus d'images = meilleure précision
2. **Fine-tuner les poids** : Ajuster le ratio LLM/MobileViT selon vos besoins
3. **Ajouter des critères** : Nouveaux critères de détection spécifiques
4. **Optimiser MobileViT** : Entraîner plus longtemps ou avec un modèle plus grand

---

## 📧 Support

Pour toute question ou problème, vérifiez :
1. `check_installation.py` - Vérifier les dépendances
2. `config.py` - Vérifier la configuration API
3. `models/mobilevit_best.pth` - Vérifier que le modèle existe

---

**Système développé avec ❤️ pour la détection d'images IA**
