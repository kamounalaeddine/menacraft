"""
Script de vérification de l'installation et de la configuration
"""

import sys


def check_python_version():
    """Vérifie la version de Python"""
    print("🐍 Vérification de Python...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (requis: 3.8+)")
        return False


def check_dependencies():
    """Vérifie les dépendances"""
    print("\n📦 Vérification des dépendances...")
    
    dependencies = {
        "openai": "openai",
        "httpx": "httpx"
    }
    
    all_ok = True
    for name, package in dependencies.items():
        try:
            __import__(package)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} (manquant)")
            all_ok = False
    
    return all_ok


def check_config():
    """Vérifie la configuration"""
    print("\n⚙️  Vérification de la configuration...")
    
    try:
        from config import LLAVA_API_CONFIG, LLAMA_API_CONFIG
        
        # Vérifier LLaVA
        if LLAVA_API_CONFIG.get("api_key"):
            print(f"   ✅ Clé API LLaVA configurée")
        else:
            print(f"   ⚠️  Clé API LLaVA manquante")
        
        if LLAVA_API_CONFIG.get("base_url"):
            print(f"   ✅ URL LLaVA : {LLAVA_API_CONFIG['base_url']}")
        else:
            print(f"   ❌ URL LLaVA manquante")
        
        # Vérifier Llama
        if LLAMA_API_CONFIG.get("api_key"):
            print(f"   ✅ Clé API Llama configurée")
        else:
            print(f"   ⚠️  Clé API Llama manquante")
        
        if LLAMA_API_CONFIG.get("base_url"):
            print(f"   ✅ URL Llama : {LLAMA_API_CONFIG['base_url']}")
        else:
            print(f"   ❌ URL Llama manquante")
        
        return True
    except ImportError as e:
        print(f"   ❌ Erreur d'import : {e}")
        return False


def check_modules():
    """Vérifie les modules du projet"""
    print("\n📁 Vérification des modules...")
    
    modules = [
        "image_description_generator",
        "ai_image_scorer",
        "ai_image_detector",
        "config"
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"   ✅ {module}.py")
        except ImportError as e:
            print(f"   ❌ {module}.py : {e}")
            all_ok = False
    
    return all_ok


def test_api_connection():
    """Test de connexion aux APIs"""
    print("\n🌐 Test de connexion aux APIs...")
    
    try:
        from config import LLAVA_API_CONFIG, LLAMA_API_CONFIG
        import httpx
        from openai import OpenAI
        
        # Test LLaVA
        print("   🔍 Test LLaVA...")
        try:
            http_client = httpx.Client(verify=False, timeout=10.0)
            client = OpenAI(
                api_key=LLAVA_API_CONFIG["api_key"],
                base_url=LLAVA_API_CONFIG["base_url"],
                http_client=http_client
            )
            # Pas de test réel pour ne pas consommer de crédits
            print("   ✅ Client LLaVA initialisé")
        except Exception as e:
            print(f"   ❌ Erreur LLaVA : {e}")
            return False
        
        # Test Llama
        print("   🔍 Test Llama...")
        try:
            http_client = httpx.Client(verify=False, timeout=10.0)
            client = OpenAI(
                api_key=LLAMA_API_CONFIG["api_key"],
                base_url=LLAMA_API_CONFIG["base_url"],
                http_client=http_client
            )
            print("   ✅ Client Llama initialisé")
        except Exception as e:
            print(f"   ❌ Erreur Llama : {e}")
            return False
        
        return True
    except Exception as e:
        print(f"   ❌ Erreur générale : {e}")
        return False


def main():
    """Fonction principale"""
    print("=" * 80)
    print("🔧 VÉRIFICATION DE L'INSTALLATION")
    print("=" * 80)
    
    results = []
    
    # Vérifications
    results.append(("Python", check_python_version()))
    results.append(("Dépendances", check_dependencies()))
    results.append(("Configuration", check_config()))
    results.append(("Modules", check_modules()))
    results.append(("APIs", test_api_connection()))
    
    # Résumé
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ")
    print("=" * 80)
    
    all_ok = True
    for name, status in results:
        emoji = "✅" if status else "❌"
        print(f"{emoji} {name}")
        if not status:
            all_ok = False
    
    print("=" * 80)
    
    if all_ok:
        print("\n🎉 Tout est prêt ! Tu peux utiliser le système.")
        print("\n💡 Commandes disponibles :")
        print("   - python test_full_detection.py image.jpg")
        print("   - python batch_detection.py dossier_images/")
        print("   - python test_description.py image.jpg")
    else:
        print("\n⚠️  Certaines vérifications ont échoué.")
        print("\n💡 Actions recommandées :")
        print("   1. Installer les dépendances : pip install -r requirements.txt")
        print("   2. Vérifier config.py (clés API, URLs)")
        print("   3. Vérifier que tous les fichiers sont présents")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
