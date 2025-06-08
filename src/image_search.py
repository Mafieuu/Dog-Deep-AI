import os
from glob import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------

def index_dataset(dataset_dir):
    """Indexe le dataset  en créant un DataFrame contenant les chemins des images et leurs classes.

    """
    data = []
    for class_dir in sorted(os.listdir(dataset_dir)):  # Trie les dossiers
        class_path = os.path.join(dataset_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_path in sorted(glob(os.path.join(class_path, '*.jpg'))):  # Trie les fichiers
            data.append({
                'image_path': img_path,
                'class': class_dir,
                'filename': os.path.basename(img_path)
            })
    return pd.DataFrame(data)

# ------------------------------------------------------------------------------------------------------

def search_by_filename(df, keyword, partial=True):
    """
        Recherche des images dans un DataFrame en fonction du nom de fichier
    """
    if partial:
        return df[df['filename'].str.contains(keyword, case=False)]
    else:
        return df[df['filename'] == keyword]

# ------------------------------------------------------------------------------------------------------

def search_by_class(df, class_name, partial=True):
    """
    Recherche des images par classe dans un DataFrame
    """
    if partial:
        return df[df['class'].str.contains(class_name, case=False)]
    else:
        return df[df['class'] == class_name]

# ------------------------------------------------------------------------------------------------------

def search_by_keyword(df, keyword):
    """
    recherche par mot-clef dans le nom du fichier ou dans le nom de la classe
    """
    return df[df.apply(lambda row: keyword.lower() in row['filename'].lower() or keyword.lower() in row['class'].lower(), axis=1)]

# ------------------------------------------------------------------------------------------------------

def show_images(df, max_images=5, figsize=(15, 5)):

    df = df.head(max_images)
    fig, axs = plt.subplots(1, len(df), figsize=figsize)
    if len(df) == 1:
        axs = [axs]
    for ax, (_, row) in zip(axs, df.iterrows()):
        img = Image.open(row['image_path'])
        ax.imshow(img)
        ax.set_title(row['class'])
        ax.axis('off')
    plt.tight_layout()
    plt.show()
