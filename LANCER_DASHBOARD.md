# Comment lancer le dashboard avec une limite de taille augmentée

Pour permettre le téléchargement de fichiers GeoTIFF volumineux (jusqu'à 500MB), suivez ces instructions :

## Option 1 : Utiliser le fichier de configuration

1. Le fichier `.streamlit/config.toml` est déjà configuré avec une limite de 500MB.
2. Lancez l'application normalement :
   ```
   streamlit run dashboard.py
   ```

## Option 2 : Configuration par ligne de commande

Si la première option ne fonctionne pas, vous pouvez spécifier la limite directement lors du démarrage :

```
streamlit run dashboard.py --server.maxUploadSize=500
```

## Option 3 : Variable d'environnement

Vous pouvez également définir une variable d'environnement :

```
# Linux/Mac
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500

# Windows PowerShell
$env:STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500

# Windows CMD
set STREAMLIT_SERVER_MAX_UPLOAD_SIZE=500

# Puis lancer l'application
streamlit run dashboard.py
```

## Traitement des fichiers très volumineux (>500MB)

Pour les fichiers extrêmement volumineux, il est recommandé de les découper en régions plus petites avant de les traiter. Utilisez l'outil `process_region.py` pour traiter des régions spécifiques :

```
python process_region.py chemin_vers_image.tif --region 1000 1000 2048 2048 --output resultat
```

Cela traitera uniquement la région spécifiée (x, y, largeur, hauteur) de l'image.
