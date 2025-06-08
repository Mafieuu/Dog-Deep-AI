import unittest
import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import sys
# Ajouter src au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.recadrer_and_save import recadrer_save 

def preparer_fichiers():
    """Crée une image avec une zone visible après recadrage et génère un fichier XML."""
    img_path = "test_image.jpg"
    annot_path = "test_annotations.xml"
    dest_dir = "output_test"

    img = Image.new("RGB", (100, 100), "blue") 
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 50, 50], fill="red")  # Ajout d'une zone rouge

    img.save(img_path)

    xml_content = """<annotation>
        <object>
            <bndbox>
                <xmin>10</xmin>
                <ymin>10</ymin>
                <xmax>50</xmax>
                <ymax>50</ymax>
            </bndbox>
        </object>
    </annotation>"""

    with open(annot_path, "w") as f:
        f.write(xml_content)

    os.makedirs(dest_dir, exist_ok=True)

    return img_path, annot_path, dest_dir

def nettoyer_fichiers(img_path, annot_path, dest_dir):
    """Supprime les fichiers créés après le test."""
    os.remove(img_path)
    os.remove(annot_path)

    for fichier in os.listdir(dest_dir):
        os.remove(os.path.join(dest_dir, fichier))

    os.rmdir(dest_dir)

class TestRecadrageImage(unittest.TestCase):
    def test_recadrage_image(self):
        """Teste le recadrage de l'image en vérifiant l'existence et les dimensions du fichier recadré."""
        img_path, annot_path, dest_dir = preparer_fichiers()
        recadrer_save(img_path, annot_path, dest_dir)

        fichier_sortie = os.path.join(dest_dir, os.path.basename(img_path))

        self.assertTrue(os.path.exists(fichier_sortie), "L'image recadrée n'a pas été sauvegardée.")

        img_recadree = Image.open(fichier_sortie)
        self.assertEqual(img_recadree.size, (40, 40), "Les dimensions de l'image recadrée sont incorrectes.")
        img_recadree.close()

        nettoyer_fichiers(img_path, annot_path, dest_dir)

if __name__ == "__main__":
    unittest.main()