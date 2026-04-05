# Installation de l'Agent de Validation

## 1. Installer les dépendances Python

```bash
pip install -r requirements.txt
```

## 2. Installer Tesseract OCR

Tesseract est nécessaire pour extraire le texte des images.

### Windows:
1. Télécharge l'installeur depuis: https://github.com/UB-Mannheim/tesseract/wiki
2. Installe Tesseract (par défaut dans `C:\Program Files\Tesseract-OCR`)
3. Ajoute le chemin au PATH ou configure dans le code

### Télécharger les données de langue pour l'arabe:
1. Va dans le dossier d'installation de Tesseract: `C:\Program Files\Tesseract-OCR\tessdata`
2. Télécharge `ara.traineddata` depuis: https://github.com/tesseract-ocr/tessdata/raw/main/ara.traineddata
3. Place le fichier dans le dossier `tessdata`

Les langues françaises et anglaises sont déjà incluses par défaut.

## 3. Utilisation

```bash
python image_logic_validator_tesseract.py
```

N'oublie pas de changer `IMAGE_PATH` dans le code avec le chemin de ton image !
