<img src="data/logo.png" width="400" height="400">


# <img src="data/doc.jpg" alt="Mon image" width="100" height="100">  DOG DEEP IA

[![ReadTheDocs](https://img.shields.io/badge/Docs-Not%20Available-red.svg)](https://readthedocs.org/)   [![Docker](https://img.shields.io/badge/Docker-%20Available-blue.svg)](https://www.docker.com/) [![Docker Compose](https://img.shields.io/badge/Docker%20Compose-%20Available-green.svg)](https://docs.docker.com/compose/)
[![Dev Containers](https://img.shields.io/badge/Dev%20Containers-%20Available-purple.svg)](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) [![Python 3.11+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)   [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0%2B-orange.svg)](https://www.tensorflow.org/)   [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)  
---

## **Classification de Chiens**



Ce projet vise à classifier différentes races de chiens à partir du dataset [Stanford Dogs ](http://vision.stanford.edu/aditya86/ImageNetDogs/). Vous y trouverez :
* L'analyse exploratoire
* Le prétraitement des images
* L’expérimentation de modèles deep learning et machine learning 
* L’évaluation et la comparaison des performances

Une documentation  sera disponible sur ReadTheDocs, et un dashboard permettra de visualiser les résultats.

---

## Table des matières

1. [Structure du projet](#structure-du-projet)
2. [Installation](#installation)
3. [Exécution (phase de test)](#exécution-phase-de-test)
4. [Dashboard & Documentation](#dashboard--documentation)

---

## Structure du projet

```text
├── data/                         
│   ├── annotation/              # Annotations pour l’entraînement
│   ├── features/                # Caractéristiques extraites (embeddings, etc.)
│   ├── images/                  # Images classées par race
│   └── lists/                   # Lists d’images pour train/test
├── notebooks_inspiration_style/ # Notebooks dont le style nous inspire
├── notebooks/                   # Notebooks du projet
├── dashboard/                   # Code du dashboard 
├── src/                         # Code python abouti,suite directe des experimentations sur notebooks
├── docs/                        # Documentation ReadTheDocs
├── Dockerfile                   # image docker
├── requirements.txt             
└── README.md                  
```

---

## Installation

> [!WARNING]
> Si vous voulez vous passez de docker rien ne garentie que l'execution se passera bien .Toute fois il vous faut installer le package pywin32 ainsi que tensorflow 2.19.0 , le plus simple serais de l'ajouter dans le requirement.txt avant de creer l'environnement virtuelle

---
### Prérequis

Assurez-vous d'avoir les éléments suivants installés sur votre machine :
- Docker Desktop  : Assurez-vous qu'il est en cours d'exécution.

- Visual Studio Code : Si vous prévoyez d'utiliser les Dev Containers.

- L'extension Dev Containers pour VS Code : Recherchez et installez "Dev Containers" dans le marketplace des extensions de VS Code.

##  Utilisation des Dev Containers ( pour les développeurs sous VS Code)

Les Dev Containers fournissent un environnement de développement directement à l'intérieur d'un conteneur Docker

### Démarrage avec les Dev Containers

1 Ouvrez le dossier du projet dans VS Code.

2 VS Code détectera automatiquement la configuration du Dev Container. Une notification apparaîtra généralement en bas à droite vous proposant de "Réouvrir dans le conteneur" (Reopen in Container).

3 Cliquez sur cette notification. Si elle n'apparaît pas, vous pouvez ouvrir la palette de commandes ( Ctrl+Shift+P ou Cmd+Shift+P ) et taper "Dev Containers: Reopen in Container".

VS Code va alors :

. Construire l'image Docker (si ce n'est pas déjà fait ou si des modifications ont été apportées au Dockerfile). Cela peut prendre quelques minutes la première fois.

. Démarrer le conteneur et y monter votre dossier de projet.

. Exécuter la commande postCreateCommand pour installer les dépendances Python (pip install -r requirements.txt).

Une fois que le conteneur est prêt, votre fenêtre VS Code sera connectée à l'environnement à l'intérieur du conteneur. Le terminal intégré de VS Code s'exécutera également dans le conteneur, et toutes les commandes Python seront exécutées avec l'interpréteur du conteneur.

> [!NOTE]
> En cas de modification du requirement.txt, l'image sera rebuild, n'incluez surtout pas tensorflow dans requirement.txt

> [!IMPORTANT]
> Le premier build  prendra de bonnes dizaines de munites à cause de tensorflow

### Accéder à Jupyter Lab
Une fois que le conteneur est lancé vous pourrez accéder à Jupyter Lab soit depuis vscode soit  depuis votre navigateur local à l'adresse :
 http://localhost:8888

### Acceder au terminale du container

Ici c'est automatique car vsCode switch directement dans le container

## Utilisation de Docker Compose

Docker-compose permet de lancer un container sans necesiter VS code
### Construire une image
Si c'est votre premiere utulisation :
``` bash
docker-compose build 
```
En cas de modification du requirement.txt  vous devez reconstruire votre image par :
```bash
docker-compose up --build
```

### Demarer les services 
```bash
docker-compose up  -d
```
le '-d' pour detacher et librer le terminal.

### Accéder à Jupyter Lab
Une fois que le conteneur est lancé vous pourrez accéder à Jupyter Lab soit depuis vscode soit  depuis votre navigateur local à l'adresse :
 http://localhost:8888

 ### Acceder au terminale du container

1. lister les containers en cour d'execution
```bash
docker ps
```
puis copierl'id du container 
2. acceder au terminal
```bash
docker exec -it <id_du_conteneur> bash
```

## Mettre à jour les dépendances : 

Si vous modifiez requirements.txt, vous devrez reconstruire votre image.

Avec Dev Containers : "Dev Containers: Rebuild Container" (depuis la palette de commandes VS Code).

Avec Docker Compose : docker compose build --no-cache puis docker compose up -d.

## Dashboard & Documentation

En construction ...
--
