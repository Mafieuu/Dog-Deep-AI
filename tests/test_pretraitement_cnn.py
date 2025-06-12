import pytest
import xml.etree.ElementTree as ET
from PIL import Image
import sys,os
# Ajouter src au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pretraitement_cnn 

@pytest.fixture
def fake_image():
    """Crée une image factice pour les tests"""
    img = Image.new("RGB", (500, 500), (255, 255, 255))  # Image blanche 500x500
    return img

@pytest.fixture
def fake_annotation(tmp_path):
    """Crée un fichier XML factice pour simuler l'annotation"""
    annotation_dir = tmp_path / "annotations"
    annotation_dir.mkdir()

    race_folder = annotation_dir / "race1"
    race_folder.mkdir()

    xml_path = race_folder / "test_image.xml"
    root = ET.Element("annotation")
    bndbox = ET.SubElement(root, "bndbox")
    ET.SubElement(bndbox, "xmin").text = "50"
    ET.SubElement(bndbox, "ymin").text = "50"
    ET.SubElement(bndbox, "xmax").text = "300"
    ET.SubElement(bndbox, "ymax").text = "300"

    tree = ET.ElementTree(root)
    tree.write(xml_path)

    return xml_path, annotation_dir

def test_recadrer_selon_bbox(fake_image, fake_annotation):
    xml_path, annotation_dir = fake_annotation
    img_path = "test_image.jpg"

    # Test de recadrage
    img_cropped = pretraitement_cnn.recadrer_selon_bbox(fake_image, img_path, str(annotation_dir), size=256, padding_color=(0,0,0))

    # Vérifie que l'image résultante a bien la taille attendue
    assert img_cropped.size == (256, 256), "L'image recadrée n'a pas la taille correcte"

    # Vérifie que l'image est bien recadrée
    cropped_box = img_cropped.getbbox()
    assert cropped_box is not None, "La bounding box ne semble pas fonctionner"