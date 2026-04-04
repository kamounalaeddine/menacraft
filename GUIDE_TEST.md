# 🧪 Guide de Test - Détecteur d'Images IA

Guide rapide pour tester le système de détection d'images IA.

---

## ✅ Vérification de l'Installation

```bash
python check_installation.py
```

**Sortie attendue :**
```
✅ Toutes les dépendances sont installées
```

---

## 🚀 Tests Rapides

### Test 1 : Image Unique

```bash
python detect_image.py "ai.jpg"
```

**Résultat attendu :**
```
✅ ANALYSE TERMINÉE
📈 Score: 0.71
💡 Conclusion: Probablement générée par IA
💾 Résultat sauvegardé: result_ai.json
```

---

### Test 2 : Image Réelle

```bash
python detect_image.py "douda.jpg"
```

**Résultat attendu :**
```
✅ ANALYSE TERMINÉE
📈 Score: 0.30
💡 Conclusion: Très probablement réelle
💾 Résultat sauvegardé: result_douda.json
```

---

### Test 3 : Depuis une URL

```bash
python detect_from_url.py "https://example.com/image.jpg"
```

---

### Test 4 : Mode Silencieux (JSON uniquement)

```bash
python detect_json_only.py "image.jpg" "mon_resultat.json"
```

**Sortie :** Affiche uniquement le chemin du fichier JSON créé

---

### Test 5 : MobileViT Seul (Plus Rapide)

```bash
python mobilevit_detector.py "dataset/ai/image.jpg"
```

**Avantages :**
- Pas besoin d'API
- Plus rapide
- Fonctionne offline

---

### Test 6 : Batch Multiple Images

```bash
python test_hybrid_simple.py
```

Teste automatiquement 3 images (2 AI + 1 réelle) et affiche un rapport.

---

## 📊 Interprétation des Résultats

### Scores

| Score | Signification |
|-------|---------------|
| **< 0.3** | ✅ Très probablement réelle |
| **0.3 - 0.5** | ✅ Probablement réelle |
| **0.5 - 0.7** | ⚠️ Incertain |
| **0.7 - 0.85** | ❌ Probablement IA |
| **> 0.85** | ❌ Très probablement IA |

### Exemples Réels

**Image AI (Score: 0.71)**
```json
{
  "scores": {
    "llm_score": 0.73,
    "mobilevit_score": 0.69,
    "final_score": 0.71
  },
  "conclusion": "Probablement générée par IA"
}
```

**Image Réelle (Score: 0.30)**
```json
{
  "scores": {
    "llm_score": 0.40,
    "mobilevit_score": 0.14,
    "final_score": 0.30
  },
  "conclusion": "Très probablement réelle"
}
```

---

## 🔍 Vérifier les Résultats JSON

### Lire un résultat

```bash
# Windows PowerShell
Get-Content result_ai.json | ConvertFrom-Json | Format-List

# Ou avec Python
python -c "import json; print(json.dumps(json.load(open('result_ai.json')), indent=2))"
```

### Extraire le score

```bash
# PowerShell
(Get-Content result_ai.json | ConvertFrom-Json).scores.final_score

# Python
python -c "import json; print(json.load(open('result_ai.json'))['scores']['final_score'])"
```

---

## 🎯 Tests de Validation

### Test sur Dataset Complet

```bash
# Tester 10 images aléatoires
python test_hybrid_simple.py
```

**Résultats attendus :**
- Précision : > 90%
- Images IA correctement détectées
- Images réelles correctement identifiées

---

## 🛠️ Dépannage

### Erreur : Image non trouvée

```bash
# Vérifier que l'image existe
ls ai.jpg

# Utiliser le chemin complet
python detect_image.py "C:/Users/fida/Desktop/hackFst/ai.jpg"
```

### Erreur : Modèle MobileViT non trouvé

```bash
# Vérifier le modèle
ls models/mobilevit_best.pth

# Si absent, entraîner le modèle
python mobilevit_trainer.py
```

### Erreur API

```bash
# Vérifier la configuration
python -c "from config import LLAVA_API_CONFIG; print(LLAVA_API_CONFIG)"

# Tester avec MobileViT seul (pas besoin d'API)
python mobilevit_detector.py "image.jpg"
```

---

## 📈 Batch Processing

### Traiter toutes les images d'un dossier

```powershell
# PowerShell
Get-ChildItem "dataset/ai/*.jpg" | ForEach-Object {
    python detect_image.py $_.FullName
}
```

### Créer un rapport CSV

```python
import json
import csv
from pathlib import Path

results = []
for json_file in Path(".").glob("result_*.json"):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        results.append({
            "image": data["image_path"],
            "score": data["scores"]["final_score"],
            "conclusion": data["conclusion"]
        })

with open('rapport.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["image", "score", "conclusion"])
    writer.writeheader()
    writer.writerows(results)

print("Rapport créé: rapport.csv")
```

---

## ✨ Exemples Avancés

### Intégration dans un script

```python
from hybrid_detector import HybridAIDetector

# Initialiser une fois
detector = HybridAIDetector()

# Analyser plusieurs images
images = ["img1.jpg", "img2.jpg", "img3.jpg"]
for img in images:
    result = detector.detect(img, save_results=False)
    score = result["scores"]["final_score"]
    
    if score > 0.7:
        print(f"⚠️ {img} est probablement IA (score: {score:.2f})")
    else:
        print(f"✅ {img} semble réelle (score: {score:.2f})")
```

### Filtrer les images IA d'un dossier

```python
from pathlib import Path
from hybrid_detector import HybridAIDetector
import shutil

detector = HybridAIDetector()

# Créer dossiers de sortie
Path("output_ai").mkdir(exist_ok=True)
Path("output_real").mkdir(exist_ok=True)

# Analyser et trier
for img in Path("mes_images").glob("*.jpg"):
    result = detector.detect(str(img), save_results=False)
    score = result["scores"]["final_score"]
    
    if score > 0.6:
        shutil.copy(img, "output_ai")
        print(f"IA: {img.name}")
    else:
        shutil.copy(img, "output_real")
        print(f"Réelle: {img.name}")
```

---

## 📝 Checklist de Test

- [ ] Installation vérifiée (`check_installation.py`)
- [ ] Test image AI (`detect_image.py "ai.jpg"`)
- [ ] Test image réelle (`detect_image.py "douda.jpg"`)
- [ ] Test depuis URL (`detect_from_url.py`)
- [ ] Test MobileViT seul (`mobilevit_detector.py`)
- [ ] Test batch (`test_hybrid_simple.py`)
- [ ] Vérification JSON (`result_*.json`)
- [ ] Scores cohérents (AI > 0.6, Réelle < 0.4)

---

## 🎓 Résultats de Validation

### Tests Effectués

| Image | Type | Score | LLM | MobileViT | Résultat |
|-------|------|-------|-----|-----------|----------|
| ai.jpg | AI | 0.71 | 0.73 | 0.69 | ✅ Correct |
| douda.jpg | Réelle | 0.30 | 0.40 | 0.14 | ✅ Correct |
| kimono_owl | AI | 0.70 | 0.50 | 1.00 | ✅ Correct |
| urban_landscape | AI | 0.78 | 0.65 | 0.98 | ✅ Correct |
| man_computer | Réelle | 0.26 | 0.40 | 0.05 | ✅ Correct |

**Précision globale : 100% (5/5)**

---

## 🚀 Prochaines Étapes

Après avoir validé le système :

1. **Tester sur vos propres images**
2. **Ajuster les seuils** si nécessaire (dans `config.py`)
3. **Entraîner MobileViT** sur votre dataset spécifique
4. **Intégrer dans votre application**

---

**Bon test ! 🎉**
