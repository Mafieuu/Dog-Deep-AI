from PIL import Image

def resize_with_padding(img, size=256, padding_color=(0, 0, 0)):
    """
    Redimensionne une image en conservant son ratio et ajoute du padding pour obtenir une image carrée.
    cela peut induire un biais mais nous allons soumettre chaque image test au même pipeline de sorte que pas de biais.

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