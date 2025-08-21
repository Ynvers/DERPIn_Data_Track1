# Dashboard de délimitation de champs agricoles

Ce projet fournit des outils pour créer un dashboard interactif permettant aux utilisateurs de sélectionner des zones d'intérêt sur des images satellite et d'y détecter les délimitations de champs agricoles à l'aide du modèle DelineateAnything.

## Structure du projet

Le projet est organisé en trois modules principaux :

1. **`image_utils.py`** : Utilitaires pour le traitement d'images satellite (chargement, normalisation, etc.)
2. **`field_detection.py`** : Implémentation de la détection de champs avec le modèle DelineateAnything
3. **`dashboard.py`** : Composants pour créer un dashboard interactif (Streamlit ou autre)

## Installation

1. Créez un environnement virtuel Python :

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

3. Suivre les instructions pour télécharger le modèle avec wget sinon le chargement direct par YOLO ne marchera pas

## Utilisation

### Mode Dashboard Streamlit

Pour lancer le dashboard interactif avec Streamlit :

```bash
streamlit run dashboard.py
```

Le dashboard vous permet de :
- Télécharger une image satellite GeoTIFF
- Sélectionner une région d'intérêt
- Détecter les délimitations de champs
- Télécharger les résultats au format GeoJSON

### Utilisation programmatique

Vous pouvez également utiliser les modules de manière programmatique :

```python
# Importer les modules
from field_detection import FieldDelineator, download_model
from image_utils import load_and_display_tci

# Télécharger le modèle
model_path = download_model()

# Initialiser le détecteur de champs
delineator = FieldDelineator(model_path)

# Traiter une image complète
tif_path = "chemin_vers_image.tif"
results = delineator.process_region(tif_path)

# Ou traiter une région spécifique
region = (1000, 1000, 2048, 2048)  # (x, y, largeur, hauteur)
results = delineator.process_region(tif_path, region=region)

print(f"Résultats sauvegardés : {results}")
```

## Architecture pour développeurs de dashboard

Le module `dashboard.py` fournit une classe `DashboardManager` qui encapsule toutes les fonctionnalités nécessaires pour un dashboard interactif. Cette classe peut être intégrée dans n'importe quelle interface web ou application de dashboard (Streamlit, Dash, etc.).

### Exemple d'intégration dans une application personnalisée :

```python
from dashboard import DashboardManager

# Initialiser le gestionnaire de dashboard
dashboard = DashboardManager()

# Charger une image
metadata = dashboard.load_image("chemin_vers_image.tif")

# Obtenir un aperçu pour affichage
overview = dashboard.get_overview_image()

# Définir une région sélectionnée par l'utilisateur
dashboard.set_region_selection(1000, 1000, 2048, 2048)

# Traiter la région sélectionnée
results = dashboard.process_selected_region()

# Nettoyer les ressources à la fin
dashboard.cleanup()
```

## Personnalisation du dashboard

Le dashboard peut être adapté à différents besoins :

1. **Utilisation avec différents frameworks** : Des exemples sont fournis pour Streamlit, mais le code peut être adapté pour Dash, Flask, etc.
2. **Personnalisation des paramètres de détection** : Vous pouvez ajuster les paramètres de détection comme la taille des tuiles, le chevauchement, les seuils de confiance, etc.
3. **Intégration de sources de données supplémentaires** : Le dashboard peut être étendu pour prendre en charge différentes sources d'images satellite.

## Notes sur les performances

- Le modèle DelineateAnything peut nécessiter un GPU pour des performances optimales
- Le traitement d'images satellite de grande taille est divisé en tuiles pour limiter la consommation de mémoire
- Les paramètres comme `tile_size` et `overlap` peuvent être ajustés selon les ressources disponibles

## Crédits

Ce projet utilise le modèle DelineateAnything développé par MykolaL : [DelineateAnything sur HuggingFace](https://huggingface.co/MykolaL/DelineateAnything)
