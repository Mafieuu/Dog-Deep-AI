import os
from glob import glob
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import numpy as np

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

# ------------------------------------------------------------------------------------------------------

def load_train_test_split_mat(
    images_dir,
    list_dir,
    train_file='train_list.mat',
    test_file='test_list.mat',
    max_races=None,
    max_images_per_race=None
):
    """
    Construit deux DataFrames (train/test) à partir des fichiers .mat et des chemins d'images.
    Garde uniquement : image_path, class, filename.
    Filtre selon max_races et max_images_per_race.
    """

    #  toutes les images depuis la structure de dossier
    df_full = index_dataset(images_dir)  

    train_mat = scipy.io.loadmat(os.path.join(list_dir, train_file))
    test_mat = scipy.io.loadmat(os.path.join(list_dir, test_file))

    # Extraire les fichiers 
    train_files = [f[0][0] if isinstance(f[0], np.ndarray) else f[0] for f in train_mat['file_list']]
    test_files = [f[0][0] if isinstance(f[0], np.ndarray) else f[0] for f in test_mat['file_list']]

    train_filenames = [os.path.basename(f) for f in train_files]
    test_filenames = [os.path.basename(f) for f in test_files]

    # Filtrage pour ne garder que les images de train et test
    df_train = df_full[df_full['filename'].isin(train_filenames)][['image_path', 'class', 'filename']].copy()
    df_test = df_full[df_full['filename'].isin(test_filenames)][['image_path', 'class', 'filename']].copy()

    # Limiter aux max_races
    if max_races is not None:
        # On trie les classes par ordre croissant, on prend les max_races premières classes
        # Ainsi autant de race dans train que dans test
        top_classes = sorted(df_train['class'].unique())[:max_races]
        df_train = df_train[df_train['class'].isin(top_classes)]
        df_test = df_test[df_test['class'].isin(top_classes)]

    # Limiter à max_images_per_race
    # Pour chaque classe, on garde au maximum max_images_per_race images
    if max_images_per_race is not None:
        df_train = df_train.groupby('class').head(max_images_per_race).reset_index(drop=True)
        df_test = df_test.groupby('class').head(max_images_per_race).reset_index(drop=True)

    return df_train, df_test

# ------------------------------------------------------------------------------------------------------
def images_to_numpy(df):
    X = []
    y = []
    for _, row in df.iterrows():
        img = Image.open(row['image_path']).convert('RGB')
        X.append(np.array(img))
        y.append(row['class'])
    return np.array(X), np.array(y)
