# -*- coding: utf-8 -*-
"""
Stanford Dogs Dataset - Utilitaires et Classes
Fichier contenant toutes les classes et fonctions importantes pour le traitement du dataset Stanford Dogs
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import xml.etree.ElementTree as ET
import scipy.io
from tqdm import tqdm
import json
import os
import gc
import time
from concurrent.futures import ThreadPoolExecutor
import pickle
from graphviz import Digraph

# Configuration GPU pour performance optimale
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})


def parse_xml_fast(xml_path):
    """
    Fonction de parsing XML rapide pour extraction des bounding boxes
    
    Args:
        xml_path: Chemin vers le fichier XML d'annotation
        
    Returns:
        dict: Dictionnaire contenant les informations de la bounding box
              (width, height, xmin, ymin, xmax, ymax) ou None si erreur
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        size = root.find('size')
        obj = root.find('object')
        bbox = obj.find('bndbox')

        return {
            'width': int(size.find('width').text),
            'height': int(size.find('height').text),
            'xmin': int(bbox.find('xmin').text),
            'ymin': int(bbox.find('ymin').text),
            'xmax': int(bbox.find('xmax').text),
            'ymax': int(bbox.find('ymax').text)
        }
    except:
        return None


class StanfordDogsProcessor:
    """
    Classe principale pour le traitement ultra-optimise du dataset Stanford Dogs
    
    Cette classe gere le chargement, le preprocessing et la creation de datasets TensorFlow
    pour le dataset Stanford Dogs avec support GPU et traitement parallele.
    """

    def __init__(self, data_root, img_size=256, use_sample=True, num_breeds=10):
        """
        Initialise le processeur Stanford Dogs
        
        Args:
            data_root: Chemin racine vers le dataset Stanford Dogs
            img_size: Taille d'image cible (carre) pour le redimensionnement
            use_sample: True pour mode echantillon, False pour dataset complet
            num_breeds: Nombre de races a utiliser si use_sample=True
        """
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.use_sample = use_sample
        self.num_breeds = num_breeds

        # Structure des repertoires du dataset Stanford Dogs
        self.images_dir = self.data_root / "Images"
        self.annotations_dir = self.data_root / "Annotation"
        self.lists_dir = self.data_root / "lists"

        # Conteneurs de donnees
        self.breed_to_id = {}  # Mapping nom de race vers ID numerique
        self.id_to_breed = {}  # Mapping ID numerique vers nom de race
        self.train_data = []   # Donnees d'entrainement
        self.test_data = []    # Donnees de test

        print(f"Mode: {'Echantillon' if use_sample else 'Complet'}")
        print(f"Races: {num_breeds if use_sample else 'Toutes (120)'}")
        print(f"Taille image: {img_size}x{img_size}")

    def load_data(self):
        """
        Chargement ultra-rapide des donnees depuis les fichiers .mat
        
        Cette methode charge les listes de fichiers, cree le mapping des races,
        et traite tous les fichiers en parallele pour extraire les informations.
        
        Returns:
            self: Retourne l'instance pour le chainage de methodes
        """
        print("\nChargement des donnees...")
        start_time = time.time()

        print("Generation des donnees...")

        # Charger les listes de fichiers depuis les fichiers .mat
        file_data = scipy.io.loadmat(self.lists_dir / "file_list.mat")
        train_data = scipy.io.loadmat(self.lists_dir / "train_list.mat")
        test_data = scipy.io.loadmat(self.lists_dir / "test_list.mat")

        # Extraire les noms de fichiers des structures MATLAB
        all_files = [f[0] for f in file_data['file_list'].flatten()]
        all_annotations = [f[0] for f in file_data['annotation_list'].flatten()]
        train_files = [f[0] for f in train_data['file_list'].flatten()]
        test_files = [f[0] for f in test_data['file_list'].flatten()]

        # Creer le mapping des races AVANT le filtrage
        self._create_breed_mapping(train_files + test_files)

        # Mode echantillon : selectionner seulement les N premieres races
        if self.use_sample:
            print(f"Selection echantillon: {self.num_breeds} races")

            # Obtenir toutes les races du dataset
            all_breeds = set()
            for file_path in train_files + test_files:
                breed = file_path.split('/')[0].split('-', 1)[1]
                all_breeds.add(breed)

            # Prendre les N premieres races (ordre alphabetique pour coherence)
            selected_breeds = sorted(list(all_breeds))[:self.num_breeds]
            selected_breeds_set = set(selected_breeds)

            # Filtrer train et test pour garder seulement les races selectionnees
            train_files = [f for f in train_files if f.split('/')[0].split('-', 1)[1] in selected_breeds_set]
            test_files = [f for f in test_files if f.split('/')[0].split('-', 1)[1] in selected_breeds_set]

            # Recreer le mapping avec seulement les races selectionnees
            self._create_breed_mapping(train_files + test_files)

            print(f"Races selectionnees: {', '.join(selected_breeds[:5])}{'...' if len(selected_breeds) > 5 else ''}")

        # Traiter les fichiers d'entrainement en parallele
        print("Traitement des fichiers d'entrainement...")
        self.train_data = self._process_file_list(train_files, all_files, all_annotations)

        # Traiter les fichiers de test en parallele
        print("Traitement des fichiers de test...")
        self.test_data = self._process_file_list(test_files, all_files, all_annotations)

        print(f"Donnees chargees en {time.time() - start_time:.2f}s")
        return self._print_stats()

    def _create_breed_mapping(self, files):
        """
        Creer le mapping bidirectionnel entre noms de races et IDs numeriques
        
        Args:
            files: Liste des chemins de fichiers pour extraire les noms de races
        """
        breeds = set()
        # Extraire les noms de races depuis les chemins de fichiers
        for file_path in files:
            breed = file_path.split('/')[0].split('-', 1)[1]
            breeds.add(breed)

        # Creer les mappings bidirectionnels tries par ordre alphabetique
        sorted_breeds = sorted(breeds)
        self.breed_to_id = {breed: i for i, breed in enumerate(sorted_breeds)}
        self.id_to_breed = {i: breed for breed, i in self.breed_to_id.items()}

    def _process_file_list(self, file_list, all_files, all_annotations):
        """
        Traiter une liste de fichiers en parallele pour extraire les informations
        
        Args:
            file_list: Liste des fichiers a traiter
            all_files: Liste complete des fichiers du dataset
            all_annotations: Liste complete des annotations correspondantes
            
        Returns:
            list: Liste des donnees traitees (dictionnaires avec infos image/annotation)
        """
        # Creer le mapping fichier vers annotation pour un acces rapide
        file_to_ann = {f: a for f, a in zip(all_files, all_annotations)}

        # Preparer les taches pour le traitement parallele
        tasks = []
        for file_path in file_list:
            if file_path in file_to_ann:
                tasks.append((file_path, file_to_ann[file_path]))

        # Traitement parallele avec ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(
                executor.map(self._process_single_file, tasks),
                total=len(tasks),
                desc="Traitement"
            ))

        # Filtrer les resultats valides (eliminer les None)
        return [r for r in results if r is not None]

    def _process_single_file(self, task):
        """
        Traiter un seul fichier image/annotation
        
        Args:
            task: Tuple (file_path, ann_path) contenant les chemins fichier et annotation
            
        Returns:
            dict: Dictionnaire avec les informations traitees ou None si erreur
        """
        file_path, ann_path = task

        # Construire les chemins complets
        breed_folder = file_path.split('/')[0]
        # Extraire le nom de fichier sans extension .jpg
        img_file_name_without_ext = file_path.split('/')[1].split('.')[0]
        img_path = self.images_dir / breed_folder / f"{img_file_name_without_ext}.jpg"
        ann_file = self.annotations_dir / ann_path

        # Verifier que le fichier image existe
        if not img_path.exists():
            print(f"Warning: Image file not found: {img_path}")
            return None

        # Parser l'annotation XML pour extraire la bounding box
        bbox = parse_xml_fast(ann_file)
        if bbox is None:
            return None

        # Extraire le nom de race et obtenir le label numerique
        breed = breed_folder.split('-', 1)[1]

        return {
            'image_path': str(img_path),
            'bbox': [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']],
            'label': self.breed_to_id[breed],
            'breed': breed
        }

    def _print_stats(self):
        """
        Afficher les statistiques du dataset charge
        
        Returns:
            self: Retourne l'instance pour le chainage de methodes
        """
        print(f"\nStatistiques:")
        print(f"   Races: {len(self.breed_to_id)}")
        print(f"   Entrainement: {len(self.train_data)}")
        print(f"   Test: {len(self.test_data)}")
        print(f"   Total: {len(self.train_data) + len(self.test_data)}")
        return self

    @tf.function
    def _preprocess_image(self, image_path, bbox):
        """
        Preprocessing optimise GPU avec tf.function pour une image
        
        Cette fonction applique le recadrage selon la bounding box,
        le redimensionnement, la normalisation et des ajustements legers.
        
        Args:
            image_path: Chemin vers l'image (tensor string)
            bbox: Bounding box [xmin, ymin, xmax, ymax] (tensor)
            
        Returns:
            tf.Tensor: Image preprocessee normalisee entre 0 et 1
        """
        # Charger l'image depuis le disque
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)

        # Recadrer selon la bounding box
        height, width = tf.shape(image)[0], tf.shape(image)[1]
        xmin, ymin, xmax, ymax = tf.unstack(bbox)

        # Securiser les coordonnees pour eviter les erreurs
        xmin = tf.maximum(0, xmin)
        ymin = tf.maximum(0, ymin)
        xmax = tf.minimum(width, xmax)
        ymax = tf.minimum(height, ymax)

        # Calculer les dimensions du crop
        crop_height = ymax - ymin
        crop_width = xmax - xmin

        # Appliquer le recadrage seulement si les dimensions sont valides
        cropped = tf.cond(
            tf.logical_and(crop_height > 10, crop_width > 10),
            lambda: tf.image.crop_to_bounding_box(image, ymin, xmin, crop_height, crop_width),
            lambda: image
        )

        # Redimensionner a la taille cible
        resized = tf.image.resize(cropped, [self.img_size, self.img_size])

        # Normaliser les valeurs entre 0 et 1
        normalized = resized / 255.0

        # Ajustements legers pour ameliorer la qualite
        #normalized = tf.image.adjust_brightness(normalized, 0.1)
        #normalized = tf.image.adjust_contrast(normalized, 1.1)

        # S'assurer que les valeurs restent dans [0, 1]
        return tf.clip_by_value(normalized, 0.0, 1.0)

    def create_dataset(self, split='train', batch_size=64, shuffle=True):
        """
        Creer un dataset TensorFlow ultra-optimise pour l'entrainement ou le test
        
        Args:
            split: 'train' ou 'test' pour selectionner les donnees
            batch_size: Taille des batches
            shuffle: True pour melanger les donnees
            
        Returns:
            tf.data.Dataset: Dataset TensorFlow pret pour l'entrainement
        """
        print(f"\nCreation du dataset {split.upper()}...")
        start_time = time.time()

        # Selectionner les donnees selon le split
        data = self.train_data if split == 'train' else self.test_data

        # Verifier que nous avons des donnees
        if len(data) == 0:
            raise ValueError(f"Aucune donnee trouvee pour le split '{split}'")

        # Extraire les composants des donnees
        image_paths = [d['image_path'] for d in data]
        bboxes = [d['bbox'] for d in data]
        labels = [d['label'] for d in data]

        # Creer le dataset TensorFlow a partir des tensors
        dataset = tf.data.Dataset.from_tensor_slices({
            'image_path': image_paths,
            'bbox': bboxes,
            'label': labels
        })

        # Melanger si necessaire (correction du bug buffer_size)
        if shuffle and len(data) > 0:
            # Utiliser la taille du dataset comme buffer_size si elle est connue
            buffer_size = min(len(data), 10000)
            dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True)

        # Preprocessing parallele avec optimisation GPU
        dataset = dataset.map(
            lambda x: {
                'image': self._preprocess_image(x['image_path'], x['bbox']),
                'label': x['label']
            },
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=False
        )

        # Filtrer les echantillons invalides
        dataset = dataset.filter(lambda x: tf.reduce_all(tf.shape(x['image']) > 0))

        # Creer les batches
        dataset = dataset.batch(batch_size, drop_remainder=True)

        # Prechargement GPU pour optimiser les performances
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        print(f"Dataset cree en {time.time() - start_time:.2f}s")
        print(f"Taille de batch: {batch_size}")

        return dataset

    def benchmark(self, dataset, name="Dataset", num_batches=20):
        """
        Benchmark de performance pour mesurer la vitesse de traitement
        
        Args:
            dataset: Dataset TensorFlow a benchmarker
            name: Nom du dataset pour l'affichage
            num_batches: Nombre de batches a tester
            
        Returns:
            float: Nombre d'echantillons traites par seconde
        """
        print(f"\nBenchmark {name.upper()}...")

        times = []
        total_samples = 0

        # Mesurer le temps de traitement sur plusieurs batches
        for i, batch in enumerate(dataset.take(num_batches)):
            start = time.time()

            # Operations GPU pour forcer le calcul
            images = batch['image']
            labels = batch['label']
            _ = tf.reduce_mean(images)  # Forcer le calcul GPU

            elapsed = time.time() - start
            times.append(elapsed)
            total_samples += tf.shape(images)[0]

            if i == 0:
                print(f"Forme du batch: {images.shape}")

        # Calculer les metriques de performance
        avg_time = np.mean(times)
        samples_per_sec = float(total_samples) / sum(times)

        print(f"Metriques de performance:")
        print(f"   Temps par batch: {avg_time:.3f}s")
        print(f"   Echantillons par seconde: {samples_per_sec:.1f}")

        # Afficher l'info memoire GPU si disponible
        try:
            gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
            print(f"   Memoire GPU: {gpu_memory['current']/(1024**3):.1f}GB")
        except:
            pass

        return samples_per_sec

    def visualize(self, dataset, num_samples=8):
        """
        Visualisation rapide des echantillons du dataset
        
        Args:
            dataset: Dataset TensorFlow a visualiser
            num_samples: Nombre d'echantillons a afficher
        """
        print("\nVisualisation des echantillons...")

        # Prendre le premier batch pour la visualisation
        for batch in dataset.take(1):
            images = batch['image'].numpy()
            labels = batch['label'].numpy()

            # Creer la grille de visualisation
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()

            # Afficher chaque echantillon
            for i in range(min(num_samples, len(images))):
                img = images[i]
                label = labels[i]
                breed = self.id_to_breed[label].replace('_', ' ')

                axes[i].imshow(img)
                axes[i].set_title(breed[:20], fontsize=10)
                axes[i].axis('off')

            plt.suptitle(f'Stanford Dogs - Mode: {"Echantillon" if self.use_sample else "Complet"} ({len(self.breed_to_id)} races)',
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()

            print(f"Stats: Forme={images.shape}, Plage=[{images.min():.3f}, {images.max():.3f}]")
            break

    def switch_mode(self, use_sample=None, num_breeds=None):
        """
        Basculer entre mode echantillon et mode complet
        
        Args:
            use_sample: True pour echantillon, False pour complet
            num_breeds: Nombre de races si mode echantillon
            
        Returns:
            self: Retourne l'instance apres rechargement des donnees
        """
        if use_sample is not None:
            self.use_sample = use_sample
        if num_breeds is not None:
            self.num_breeds = num_breeds

        print(f"Mode change: {'Echantillon' if self.use_sample else 'Complet'}")
        if self.use_sample:
            print(f"Races: {self.num_breeds}")
        return self.load_data()


def run_pipeline(data_root, use_sample=True, num_breeds=10, img_size=256):
    """
    Pipeline principal ultra-optimise pour le traitement du dataset Stanford Dogs
    
    Cette fonction execute le pipeline complet : chargement, preprocessing,
    creation des datasets, benchmark et visualisation.
    
    Args:
        data_root: Chemin racine vers le dataset
        use_sample: True pour mode echantillon, False pour dataset complet
        num_breeds: Nombre de races si mode echantillon
        img_size: Taille d'image cible
        
    Returns:
        tuple: (processeur, dataset_train, dataset_test)
    """
    print("Pipeline Stanford Dogs - Ultra-Optimise")
    print("=" * 60)

    # Initialiser le processeur avec les parametres specifies
    processor = StanfordDogsProcessor(
        data_root=data_root,
        img_size=img_size,
        use_sample=use_sample,
        num_breeds=num_breeds
    )

    # Charger les donnees depuis les fichiers .mat
    processor.load_data()

    # Creer les datasets avec des tailles de batch optimales pour GPU
    batch_size = 128

    print(f"\nTaille de batch optimisee: {batch_size}")

    # Creer les datasets d'entrainement et de test
    train_dataset = processor.create_dataset('train', batch_size=batch_size)
    test_dataset = processor.create_dataset('test', batch_size=batch_size//2)

    # Benchmark de performance pour mesurer les vitesses
    train_speed = processor.benchmark(train_dataset, "Entrainement")
    test_speed = processor.benchmark(test_dataset, "Test")

    # Visualisation des echantillons
    processor.visualize(train_dataset)

    print(f"\nPipeline termine")
    print(f"Performance entrainement: {train_speed:.1f} echantillons/sec")
    print(f"Performance test: {test_speed:.1f} echantillons/sec")

    return processor, train_dataset, test_dataset


def show_pipeline_graph():
    """
    Afficher un diagramme du pipeline de traitement
    
    Returns:
        Digraph: Objet graphique representant le pipeline
    """
    dot = Digraph(comment='Stanford Dogs Pipeline', format='png')
    dot.attr(rankdir='LR', size='10')

    # Definir les noeuds du pipeline
    dot.node('A', 'Chargement .mat\n(file_list, train_list, test_list)', shape='box')
    dot.node('B', 'Mapping race ↔ ID\n(_create_breed_mapping)', shape='box')
    dot.node('C', 'Filtrage (mode echantillon)', shape='box')
    dot.node('D', 'Traitement fichiers\n(_process_file_list)', shape='box')
    dot.node('E', 'Traitement unitaire\n(_process_single_file)', shape='box')
    dot.node('F', 'Preprocessing image\n(_preprocess_image)', shape='box')
    dot.node('G', 'Dataset TensorFlow\n(create_dataset)', shape='box')
    dot.node('H', 'Visualisation / Benchmark', shape='ellipse')

    # Definir les connexions entre les etapes
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    dot.edge('G', 'H')

    return dot


def save_processor_data(processor, output_dir, mode_suffix="sample"):
    """
    Sauvegarder les donnees du processeur dans des fichiers pickle
    
    Args:
        processor: Instance de StanfordDogsProcessor
        output_dir: Repertoire de sortie
        mode_suffix: Suffixe pour identifier le mode (sample/full)
    """
    output_dir = Path(output_dir)
    train_output_dir = output_dir / "data_train"
    test_output_dir = output_dir / "data_test"

    # Creer les repertoires de sortie s'ils n'existent pas
    train_output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir.mkdir(parents=True, exist_ok=True)

    # Definir les chemins de fichiers pour la sauvegarde
    train_data_file = train_output_dir / f"train_data_{mode_suffix}.pkl"
    test_data_file = test_output_dir / f"test_data_{mode_suffix}.pkl"
    processor_file = output_dir / f"processor_object_{mode_suffix}.pkl"

    # Sauvegarder les donnees d'entrainement et de test
    with open(train_data_file, 'wb') as f:
        pickle.dump(processor.train_data, f)

    with open(test_data_file, 'wb') as f:
        pickle.dump(processor.test_data, f)

    # Sauvegarder l'objet processeur complet
    try:
        with open(processor_file, 'wb') as f:
            pickle.dump(processor, f)
        print(f"Objet processor sauvegarde a: {processor_file}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde de l'objet processor: {e}")

    print(f"Train data saved to: {train_data_file}")
    print(f"Test data saved to: {test_data_file}")