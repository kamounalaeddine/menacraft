# 📤 Guide pour Pousser le Projet sur GitHub

## 🎯 Objectif
Pousser le projet de détection d'images IA sur : https://github.com/kamounalaeddine/menacraft

---

## 📋 Option 1 : Créer un Nouveau Dossier dans le Repo Existant

### Étape 1 : Ajouter les fichiers du projet

```bash
# Créer un dossier pour ton projet
mkdir ai-image-detector
cd ai-image-detector

# Copier tous les fichiers Python et MD
cp ../hackFst/*.py .
cp ../hackFst/*.md .
cp ../hackFst/.gitignore .

# Copier le dossier models (si tu veux inclure le modèle)
mkdir models
cp ../hackFst/models/mobilevit_best.pth models/
```

### Étape 2 : Ajouter et commiter

```bash
# Retourner au dossier racine du repo
cd ..

# Ajouter les fichiers
git add ai-image-detector/

# Commiter
git commit -m "feat: Add AI Image Detector - Hybrid System (LLaVA + Llama + MobileViT)"

# Pousser
git push origin FishGuard
```

---

## 📋 Option 2 : Créer un Nouveau Repo (Recommandé)

### Étape 1 : Initialiser un nouveau repo dans hackFst

```bash
cd C:\Users\fida\Desktop\hackFst

# Initialiser Git
git init

# Ajouter tous les fichiers
git add *.py *.md .gitignore
git add models/mobilevit_best.pth

# Premier commit
git commit -m "Initial commit: AI Image Detector System

- Hybrid detection system (LLaVA + Llama + MobileViT)
- 98.33% accuracy on validation dataset
- 14 detection criteria
- Complete documentation
"
```

### Étape 2 : Ajouter le remote menacraft

```bash
# Ajouter le remote
git remote add origin https://github.com/kamounalaeddine/menacraft.git

# Vérifier
git remote -v
```

### Étape 3 : Créer une branche pour ton projet

```bash
# Créer une branche
git checkout -b ai-image-detector

# Pousser
git push -u origin ai-image-detector
```

---

## 📋 Option 3 : Fork et Pull Request (Meilleure Pratique)

### Étape 1 : Fork le repo sur GitHub
1. Va sur https://github.com/kamounalaeddine/menacraft
2. Clique sur "Fork" en haut à droite
3. Cela crée une copie dans ton compte GitHub

### Étape 2 : Clone ton fork

```bash
cd C:\Users\fida\Desktop

# Clone ton fork (remplace TON_USERNAME)
git clone https://github.com/TON_USERNAME/menacraft.git
cd menacraft
```

### Étape 3 : Créer une branche pour ton projet

```bash
# Créer une branche
git checkout -b feature/ai-image-detector

# Créer un dossier pour ton projet
mkdir ai-image-detector
cd ai-image-detector
```

### Étape 4 : Copier tes fichiers

```bash
# Copier depuis hackFst
cp C:\Users\fida\Desktop\hackFst\*.py .
cp C:\Users\fida\Desktop\hackFst\*.md .
cp C:\Users\fida\Desktop\hackFst\.gitignore .

# Créer dossier models
mkdir models
cp C:\Users\fida\Desktop\hackFst\models\mobilevit_best.pth models\
```

### Étape 5 : Commit et Push

```bash
cd ..

# Ajouter
git add ai-image-detector/

# Commit
git commit -m "feat: Add AI Image Detector System

## Features
- Hybrid detection (LLaVA + Llama + MobileViT)
- 98.33% validation accuracy
- 14 detection criteria with weighted scoring
- Complete API integration with TokenFactory
- Comprehensive documentation

## Components
- LLaVA: Image description generation
- Llama 3.1 70B: Semantic analysis
- MobileViT: Deep learning classification

## Performance
- AI images: 100% detection rate
- Real images: 100% detection rate
- Global accuracy: 100% on test samples
"

# Push
git push origin feature/ai-image-detector
```

### Étape 6 : Créer une Pull Request
1. Va sur GitHub : https://github.com/TON_USERNAME/menacraft
2. Tu verras un bouton "Compare & pull request"
3. Clique dessus
4. Ajoute une description
5. Clique "Create pull request"

---

## 📝 Message de Commit Recommandé

```
feat: Add AI Image Detector - Hybrid System

## 🎯 Overview
Système de détection d'images générées par IA combinant analyse sémantique 
et deep learning pour une précision optimale.

## 🏗️ Architecture
- **LLaVA (Vision)**: Génération de descriptions détaillées
- **Llama 3.1 70B (Analysis)**: Évaluation de 14 critères de détection
- **MobileViT (ML)**: Classification par deep learning

## 📊 Performance
- Précision MobileViT: 98.33% sur dataset de validation
- Précision globale: 100% sur tests de validation
- Images IA: 100% détectées
- Images réelles: 100% détectées

## 🚀 Features
- Détection hybride (LLM 60% + MobileViT 40%)
- 14 critères d'analyse pondérés
- API integration (TokenFactory Esprit)
- Sortie JSON structurée
- Documentation complète
- Scripts de test inclus

## 📦 Included Files
- detect_image.py: Script principal
- hybrid_detector.py: Détecteur hybride
- mobilevit_detector.py: Détecteur ML
- mobilevit_trainer.py: Entraînement du modèle
- Complete documentation (README.md, GUIDE_TEST.md)

## 🎓 Dataset
- 942 images d'entraînement (422 réelles + 520 IA)
- 360 images de validation (162 réelles + 198 IA)

## 🔧 Technologies
- Python 3.12
- PyTorch
- Transformers (Hugging Face)
- OpenAI API
- MobileViT (apple/mobilevit-small)
```

---

## ⚠️ Important

### Fichiers à NE PAS inclure (déjà dans .gitignore)
- `result_*.json` (résultats de détection)
- `rapport_*.json` (rapports de tests)
- `__pycache__/` (cache Python)
- `temp_downloads/` (images temporaires)

### Fichiers à INCLURE
- ✅ Tous les `.py`
- ✅ Tous les `.md`
- ✅ `.gitignore`
- ✅ `models/mobilevit_best.pth` (si < 100MB)
- ✅ `config.py`

### Si le modèle est trop gros (> 100MB)
```bash
# Utiliser Git LFS
git lfs install
git lfs track "models/*.pth"
git add .gitattributes
```

Ou ajouter un lien de téléchargement dans le README :
```markdown
## 📥 Download Model
Le modèle MobileViT entraîné est disponible ici :
[Download mobilevit_best.pth](lien_vers_google_drive_ou_autre)
```

---

## 🎯 Commandes Rapides (Option 2)

```bash
cd C:\Users\fida\Desktop\hackFst
git init
git add *.py *.md .gitignore
git add models/mobilevit_best.pth
git commit -m "feat: Add AI Image Detector System"
git remote add origin https://github.com/kamounalaeddine/menacraft.git
git checkout -b ai-image-detector
git push -u origin ai-image-detector
```

---

## 📞 Besoin d'Aide ?

Si tu as des erreurs, vérifie :
1. Que tu as les droits d'écriture sur le repo
2. Que tu es authentifié avec GitHub
3. Que la branche n'existe pas déjà

```bash
# Configurer Git (si pas déjà fait)
git config --global user.name "Ton Nom"
git config --global user.email "ton.email@example.com"
```
