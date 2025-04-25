
# <img src="data/doc.jpg" alt="Mon image" width="100" height="100">  Readme principale

* pas de pip install dans un notebook, utiliser a la place un venv et un conda env

# Classification de Chiens 
  
Ce projet vise à classifier différentes races de chiens issues du [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/).  

Il s'agira de :  
- Prétraiter les images 
- Expérimenter avec différentes modéles des modèle de deep learning/machine learning  adapté à la classification d'images 
- Évaluer la performance des modèle  

Ce projet sera accompagné d'un dashboard [...], un ReadTheDocs  spécialement conçu pour les étudiants qui découvrent la classification d’images et l’apprentissage automatique avec le dataset Stanford Dogs Dataset . Cette doc expliquera les concepts, les étapes du projet et les bonnes pratiques pour reproduire et approfondir l’expérimentation.

# Comment éxécuter ce projet ?
Télechargez le dataset http://vision.stanford.edu/aditya86/ImageNetDogs/ puis decompressez les fichiers et placez les comme suite :
## 📂 annotation
Contient les fichiers d'annotation utilisés pour l'entraînement du modèle.
## 📂 features
Stocke les caractéristiques extraites des données brutes

## 📂 images
Dossier contenant les images utilisées de chiens.Chaque race a son sous-dossier comme dans le dataset

## 📂 lists
Comme dans le datset
## ⚙️ Environnement Condas
Ce projet utulise un conda env pour les notebookss ,pour le recréer, exécutez :
```gitbash
conda env create -f environment.yaml

