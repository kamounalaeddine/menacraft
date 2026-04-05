# Image Content Analyzer - Contextual Analysis

Advanced image text extraction and contextual analysis system using OCR.space, LLaVA, and Llama 3.1 70B.

## Features

- **Multi-source OCR**: OCR.space API for text extraction (supports Arabic, French, English)
- **Visual Reading**: LLaVA for complete visual text reading (handles cursive, stylized fonts)
- **Contextual Analysis**: Deep contextual understanding with Llama 3.1 70B
- **Smart Highlighting**: Visual feedback for content coherence issues

## What's New

This version focuses on **contextual analysis** rather than word-level error detection:

- ✅ Comprehensive content type analysis
- ✅ Audience targeting identification
- ✅ Tone and style detection
- ✅ Key message extraction
- ✅ Usage context identification
- ❌ No more word-level error lists (mots_problematiques removed)

## Installation

```bash
pip install -r requirements.txt
```

See `INSTALLATION.md` for detailed setup instructions.

## Usage

### Basic Usage

```python
from image_content_analyzer import ImageContentAnalyzer

# Initialize analyzer
analyzer = ImageContentAnalyzer(
    api_key="your-api-key",
    ocr_space_key="your-ocr-space-key",
    base_url="https://tokenfactory.esprit.tn/api"
)

# Analyze image
result = analyzer.analyze_image_complete("image.jpg")

# Access results
print(f"Type: {result['description_avancee']['type_contenu']}")
print(f"Objective: {result['description_avancee']['objectif']}")
print(f"Tone: {result['description_avancee']['ton_style']}")
print(f"Key Message: {result['description_avancee']['message_cle']}")
print(f"Context: {result['description_avancee']['contexte_utilisation']}")
```

### Command Line

```bash
python image_content_analyzer.py
```

Update `IMAGE_PATH` in the script to analyze your image.

## Output Structure

```json
{
  "texte_ocr_space": "Raw OCR text...",
  "texte_llava": "LLaVA visual reading...",
  "texte_utilise": "Best text source...",
  "texte_reconstruit": "Corrected text...",
  "description_avancee": {
    "type_contenu": "motivationnel",
    "objectif": "motiver",
    "audience_cible": "grand public",
    "ton_style": "inspirant",
    "message_cle": "Key message summary",
    "contexte_utilisation": "réseaux sociaux, affiche, site web",
    "resume": "Detailed content description..."
  },
  "langue": "anglais",
  "est_logique": "OUI",
  "explication": "Analysis explanation...",
  "source_principale": "LLaVA",
  "image_highlighted": "image_highlighted.jpg"
}
```

## Pipeline

1. **OCR.space API** → Text extraction (fallback/complement)
2. **LLaVA** → Complete visual reading (primary source)
3. **Llama 3.1 70B** → Deep contextual analysis
4. **PIL** → Contextual highlighting (if needed)

## Configuration

Key parameters in the code:
- `api_key`: Your API key for LLaVA and Llama
- `ocr_space_key`: OCR.space API key
- `base_url`: API endpoint
- `llava_model`: Vision model for text reading
- `llama_model`: Language model for analysis

## Requirements

- Python 3.8+
- PIL/Pillow
- httpx
- openai
- requests

## License

MIT
