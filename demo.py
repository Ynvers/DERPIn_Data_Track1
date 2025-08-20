"""
Script de démonstration pour le traitement d'une image satellite et la détection de champs.
Ce script illustre l'utilisation des modules image_utils.py et field_detection.py
pour traiter une image satellite et détecter les délimitations de champs.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from image_utils import load_and_display_tci, download_tci_image
from field_detection import FieldDelineator, download_model


def main():
    """Fonction principale de démonstration"""
    
    print("🚀 Démonstration de détection de champs sur image satellite")
    
    # 1. Télécharger une image TCI exemple si nécessaire
    tci_url = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/54/T/WN/2023/8/S2A_54TWN_20230815_0_L2A/TCI.tif"
    tci_path = "TCI_demo.tif"
    
    try:
        tci_path = download_tci_image(tci_url, tci_path)
        
        # 2. Afficher l'image
        print("\n📊 Affichage de l'image satellite...")
        rgb_array, crs, transform = load_and_display_tci(tci_path)
        
        # 3. Télécharger le modèle DelineateAnything
        print("\n🧠 Préparation du modèle...")
        model_path = download_model()
        
        if model_path is None:
            print("❌ Erreur: Impossible de télécharger le modèle.")
            return
        
        # 4. Initialiser le détecteur de champs
        detector = FieldDelineator(model_path)
        
        # 5. Sélection d'une région (par défaut : 2048×2048 pixels au centre de l'image)
        h, w = rgb_array.shape[:2]
        region = (w//2 - 1024, h//2 - 1024, 2048, 2048)  # (x, y, width, height)
        print(f"\n🔍 Région sélectionnée: {region}")
        
        # 6. Traiter la région et détecter les champs
        print("\n🔍 Lancement de la détection de champs...")
        results = detector.process_region(
            tci_path, 
            region=region,
            output_prefix="demo_output",
            tile_size=1024,
            overlap=128
        )
        
        # 7. Afficher le résultat
        if results:
            from PIL import Image
            print("\n🖼️ Affichage du résultat...")
            overlay = Image.open(results["overlay"])
            
            plt.figure(figsize=(12, 8))
            plt.imshow(overlay)
            plt.title("Délimitation des champs")
            plt.axis('off')
            plt.show()
            
            print(f"\n✅ Traitement terminé. Fichiers générés:")
            print(f"   - {results['overlay']} : Image avec délimitations")
            print(f"   - {results['geojson']} : Délimitations au format GeoJSON")
        else:
            print("❌ Erreur lors du traitement.")
    
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()
