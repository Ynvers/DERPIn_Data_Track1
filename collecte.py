import requests
import json
from pathlib import Path
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://www.aagwa.org/api"

out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

# Créer les sous-dossiers principaux
predictions_dir = out_dir / "predictions"
crop_mappings_dir = out_dir / "crop-mappings"
predictions_dir.mkdir(exist_ok=True)
crop_mappings_dir.mkdir(exist_ok=True)

def download_tif(url: str, dest_path: Path):
    """Télécharge un GeoTIFF avec vérifications"""
    try:
        logger.info(f"Downloading {url} -> {dest_path}")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        
        # Vérifier le type de contenu
        content_type = r.headers.get('content-type', '')
        if 'text/html' in content_type:
            logger.warning(f"❌ HTML content received instead of GeoTIFF: {url}")
            return False
            
        # Créer le dossier parent
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Télécharger le fichier
        total_size = int(r.headers.get('content-length', 0))
        with open(dest_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\r{dest_path.name}: {progress:.1f}%", end="", flush=True)
            print()  # Nouvelle ligne
        
        logger.info(f"✅ Successfully downloaded: {dest_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error downloading {url}: {e}")
        return False

def fetch_all_predictions():
    """Récupère toutes les prédictions disponibles dans l'API"""
    logger.info("🔍 Fetching all predictions from API...")
    
    try:
        response = requests.get(f"{BASE_URL}/predictions", timeout=60)
        response.raise_for_status()
        
        predictions = response.json()
        logger.info(f"✅ Found {len(predictions)} predictions")
        
        # Sauvegarder toutes les prédictions dans un JSON
        predictions_file = out_dir / "all_predictions.json"
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 All predictions saved to: {predictions_file}")
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Error fetching predictions: {e}")
        return []

def fetch_all_crop_mappings():
    """Récupère toutes les cartes de cultures disponibles dans l'API"""
    logger.info("🔍 Fetching all crop mappings from API...")
    
    try:
        response = requests.get(f"{BASE_URL}/crop-mappings", timeout=60)
        response.raise_for_status()
        
        crop_mappings = response.json()
        logger.info(f"✅ Found {len(crop_mappings)} crop mappings")
        
        # Sauvegarder toutes les cartes de cultures dans un JSON
        mappings_file = out_dir / "all_crop_mappings.json"
        with open(mappings_file, 'w', encoding='utf-8') as f:
            json.dump(crop_mappings, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 All crop mappings saved to: {mappings_file}")
        return crop_mappings
        
    except Exception as e:
        logger.error(f"❌ Error fetching crop mappings: {e}")
        return []

def download_predictions(predictions_list):
    """Télécharge tous les fichiers de prédictions"""
    logger.info(f"⬇️  Starting download of {len(predictions_list)} predictions...")
    
    successful_downloads = 0
    failed_downloads = 0
    
    for item in predictions_list:
        try:
            # Vérifier que les champs requis existent
            if "file" not in item or "id" not in item:
                logger.warning(f"❌ Missing file or id in item: {item}")
                failed_downloads += 1
                continue
            
            # Construire l'URL complète à partir du champ "file"  
            tif_url = f"https://www.aagwa.org/{item['file'].lstrip('/')}"
            
            # Nom de fichier basé sur l'ID
            filename = f"{item['id']}.tif"
            dest_path = predictions_dir / filename
            
            if download_tif(tif_url, dest_path):
                successful_downloads += 1
            else:
                failed_downloads += 1
                
        except Exception as e:
            logger.error(f"❌ Exception downloading prediction {item.get('id', 'unknown')}: {e}")
            failed_downloads += 1
    
    logger.info(f"📊 Predictions download: {successful_downloads} successful, {failed_downloads} failed")
    return successful_downloads, failed_downloads

def download_crop_mappings(mappings_list):
    """Télécharge tous les fichiers de cartes de cultures"""
    logger.info(f"⬇️  Starting download of {len(mappings_list)} crop mappings...")
    
    successful_downloads = 0
    failed_downloads = 0
    
    for item in mappings_list:
        try:
            # Vérifier que les champs requis existent
            if "file" not in item or "id" not in item:
                logger.warning(f"❌ Missing file or id in item: {item}")
                failed_downloads += 1
                continue
            
            # Construire l'URL complète à partir du champ "file"
            tif_url = f"https://www.aagwa.org/{item['file'].lstrip('/')}"
            
            # Nom de fichier basé sur l'ID
            filename = f"{item['id']}.tif"
            dest_path = crop_mappings_dir / filename
            
            if download_tif(tif_url, dest_path):
                successful_downloads += 1
            else:
                failed_downloads += 1
                
        except Exception as e:
            logger.error(f"❌ Exception downloading crop mapping {item.get('id', 'unknown')}: {e}")
            failed_downloads += 1
    
    logger.info(f"📊 Crop mappings download: {successful_downloads} successful, {failed_downloads} failed")
    return successful_downloads, failed_downloads

# ====== SCRIPT PRINCIPAL ======
if __name__ == "__main__":
    logger.info("🚀 Starting Climate Smart Agriculture data collection...")
    
    # 1. Récupérer toutes les prédictions
    all_predictions = fetch_all_predictions()
    
    # 2. Récupérer toutes les cartes de cultures  
    all_crop_mappings = fetch_all_crop_mappings()
    
    # 3. Télécharger les fichiers de prédictions
    pred_success, pred_failed = download_predictions(all_predictions)
    
    # 4. Télécharger les fichiers de cartes de cultures
    map_success, map_failed = download_crop_mappings(all_crop_mappings)
    
    # 5. Créer un résumé du dataset collecté
    dataset_summary = {
        "collection_date": datetime.now().isoformat(),
        "collection_timestamp": datetime.now().timestamp(),
        "api_base_url": BASE_URL,
        "total_predictions_available": len(all_predictions),
        "total_crop_mappings_available": len(all_crop_mappings),
        "download_results": {
            "predictions": {
                "successful": pred_success,
                "failed": pred_failed,
                "success_rate": pred_success / (pred_success + pred_failed) * 100 if (pred_success + pred_failed) > 0 else 0
            },
            "crop_mappings": {
                "successful": map_success,
                "failed": map_failed,
                "success_rate": map_success / (map_success + map_failed) * 100 if (map_success + map_failed) > 0 else 0
            }
        },
        "data_structure": {
            "predictions_folder": str(predictions_dir),
            "crop_mappings_folder": str(crop_mappings_dir),
            "naming_convention": "Files are named with their API ID (e.g., {id}.tif)",
            "metadata_files": ["all_predictions.json", "all_crop_mappings.json"]
        }
    }
    
    # Sauvegarder le résumé
    summary_file = out_dir / "dataset_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_summary, f, indent=2, ensure_ascii=False)
    
    # Résumé final
    logger.info("🎉 Data collection completed!")
    logger.info(f"📊 Total predictions: {pred_success}/{len(all_predictions)} downloaded")
    logger.info(f"📊 Total crop mappings: {map_success}/{len(all_crop_mappings)} downloaded")
    logger.info(f"📁 Dataset structure:")
    logger.info(f"  ├── {predictions_dir} ({pred_success} files)")
    logger.info(f"  ├── {crop_mappings_dir} ({map_success} files)")
    logger.info(f"  ├── all_predictions.json ({len(all_predictions)} items)")
    logger.info(f"  ├── all_crop_mappings.json ({len(all_crop_mappings)} items)")
    logger.info(f"  └── dataset_summary.json")
    logger.info(f"📋 Complete summary saved to: {summary_file}")
