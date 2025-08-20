"""
Utilitaire pour traiter des zones spécifiques d'une image satellite.

Ce script permet de sélectionner et traiter des coordonnées spécifiques
d'une image GeoTIFF pour la détection de champs agricoles.
"""

import argparse
import sys
import os
from pathlib import Path

# Importer les modules locaux
try:
    from field_detection import FieldDelineator, download_model
except ImportError:
    print("❌ Erreur: Modules requis non trouvés.")
    print("📝 Assurez-vous d'avoir installé les dépendances:")
    print("   pip install -r requirements_dashboard.txt")
    sys.exit(1)


def parse_args():
    """Parse les arguments de la ligne de commande"""
    parser = argparse.ArgumentParser(
        description="Outil de détection de champs dans une image satellite."
    )
    parser.add_argument(
        "tif_path",
        help="Chemin vers l'image satellite GeoTIFF"
    )
    parser.add_argument(
        "--region", "-r",
        nargs=4, type=int, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
        help="Région à traiter (x y width height). Si non spécifié, traite l'image entière."
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Préfixe pour les fichiers de sortie"
    )
    parser.add_argument(
        "--tile-size", "-t",
        type=int, default=1024,
        help="Taille des tuiles pour le traitement par morceaux"
    )
    parser.add_argument(
        "--overlap", "-v",
        type=int, default=128,
        help="Chevauchement entre les tuiles"
    )
    parser.add_argument(
        "--confidence", "-c",
        type=float, default=0.35,
        help="Seuil de confiance pour la détection (0.0-1.0)"
    )
    parser.add_argument(
        "--iou", "-i",
        type=float, default=0.5,
        help="Seuil IoU pour la suppression des non-maximums (0.0-1.0)"
    )
    parser.add_argument(
        "--model", "-m",
        help="Chemin vers le modèle DelineateAnything.pt"
    )
    
    return parser.parse_args()


def main():
    """Fonction principale"""
    args = parse_args()
    
    # Vérifier que l'image existe
    if not Path(args.tif_path).exists():
        print(f"❌ Erreur: Le fichier {args.tif_path} n'existe pas.")
        return 1
    
    # Télécharger ou utiliser le modèle spécifié
    model_path = args.model if args.model else download_model()
    if not model_path:
        print("❌ Erreur: Impossible d'accéder au modèle.")
        return 1
    
    # Initialiser le détecteur
    detector = FieldDelineator(model_path)
    
    # Définir la région
    region = args.region if args.region else None
    
    # Afficher les informations de traitement
    print(f"🛰️ Image: {args.tif_path}")
    if region:
        print(f"🔍 Région: ({region[0]}, {region[1]}, {region[2]}, {region[3]})")
    else:
        print("🔍 Région: Image entière")
    print(f"⚙️ Taille de tuile: {args.tile_size}, Chevauchement: {args.overlap}")
    print(f"📊 Seuil de confiance: {args.confidence}, Seuil IoU: {args.iou}")
    print(f"💾 Sortie: {args.output}_overlay.png, {args.output}_fields.geojson")
    
    # Lancer le traitement
    print("\n🚀 Lancement de la détection...")
    results = detector.process_region(
        args.tif_path,
        region=region,
        output_prefix=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        conf_thr=args.confidence,
        iou_thr=args.iou
    )
    
    if results:
        print(f"\n✅ Traitement terminé avec succès!")
        print(f"   Résultats sauvegardés:")
        print(f"   - Image avec délimitations: {results['overlay']}")
        print(f"   - Fichier GeoJSON: {results['geojson']}")
        return 0
    else:
        print("❌ Erreur lors du traitement.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
