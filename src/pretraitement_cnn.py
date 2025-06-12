from PIL import Image
import os
import xml.etree.ElementTree as ET


def resize_with_padding(img, size=256, padding_color=(0, 0, 0)):
    """
    Redimensionne une image en conservant son ratio et ajoute du padding pour obtenir une image carrée.
    cela peut induire un biais mais nous allons soumettre chaque image test au même pipeline de sorte que pas de biais.
    - Si l'image image fait 300x150 pixels, son ratio est 2:1
    - Après redimensionnement, elle devient 256x128 pixels (même ratio 2:1).
    - Pour obtenir 256x256, on ajoute du padding .


    """
    
    # Récupérer les dimensions de l’image originale
    w, h = img.size  

    # Calcul du facteur d’échelle pour ajuster le plus grand côté à 'size'
    scale = size / max(w, h)
    
    # Redimensionner l’image en conservant son ratio
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Création d’un fond carré pour le padding
    new_img = Image.new("RGB", (size, size), padding_color)
    
    # Calcul des coordonnées pour centrer l’image redimensionnée
    x_offset = (size - img.size[0]) // 2
    y_offset = (size - img.size[1]) // 2

    # Coller l’image redimensionnée sur le fond carré
    new_img.paste(img, (x_offset, y_offset))

    return new_img

def recadrer_selon_bbox(img, img_path, annotation_dir, size=256, padding_color=(0,0,0)):

    """
    Cherche l'annotation XML correspondant à l'image, découpe selon la bounding box, puis resize avec resize_with_padding()
    """
    base = os.path.splitext(os.path.basename(img_path))[0]

    for race_folder in os.listdir(annotation_dir):
        race_path = os.path.join(annotation_dir, race_folder)
        if not os.path.isdir(race_path):
            continue
        # Recherche partielle du fichier d'annotation
        for fname in os.listdir(race_path):
            if base in fname:
                annotation_path = os.path.join(race_path, fname)
                break
        else:
            continue
        break
    else:
        raise FileNotFoundError(f"Annotation pour {base} non trouvée dans {annotation_dir}")

    # Parse le XML pour extraire la bounding box
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    bndbox = root.find(".//bndbox")
    xmin = int(bndbox.find("xmin").text)
    ymin = int(bndbox.find("ymin").text)
    xmax = int(bndbox.find("xmax").text)
    ymax = int(bndbox.find("ymax").text)

    # Découpe l'image
    img_cropped = img.crop((xmin, ymin, xmax, ymax))
    # Resize avec padding
    img_final = resize_with_padding(img_cropped, size=size, padding_color=padding_color)
    return img_final