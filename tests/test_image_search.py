import unittest
import os
import pandas as pd
import sys
from unittest.mock import patch
from PIL import Image 

# Ajouter src au chemin de recherche des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import image_search

# ------------------------------------------------------------------------------------------------------------

class TestIndexDataset(unittest.TestCase):
    def setUp(self):
        """ Crée un dataset avec plusieurs races de chiens """
        self.test_dir = "../data/"
        os.makedirs(os.path.join(self.test_dir, "GoldenRetriever"), exist_ok=True)
        os.makedirs(os.path.join(self.test_dir, "Bulldog"), exist_ok=True)

        with open(os.path.join(self.test_dir, "GoldenRetriever/golden1.jpg"), "w") as f:
            f.write("")
        with open(os.path.join(self.test_dir, "Bulldog/bulldog1.jpg"), "w") as f:
            f.write("")

    def tearDown(self):
        """ Supprime les fichiers créés pour le test """
        os.remove(os.path.join(self.test_dir, "GoldenRetriever/golden1.jpg"))
        os.remove(os.path.join(self.test_dir, "Bulldog/bulldog1.jpg"))
        os.rmdir(os.path.join(self.test_dir, "GoldenRetriever"))
        os.rmdir(os.path.join(self.test_dir, "Bulldog"))

    def test_index_dataset(self):
            df = image_search.index_dataset(self.test_dir)
            self.assertFalse(df.empty)  # Vérifie que le dataset n'est pas vide
            self.assertIn("GoldenRetriever", df["class"].values)
            self.assertIn("Bulldog", df["class"].values)
            
            #Vérifiez que les fichiers sont présents, peu importe l'ordre.
            self.assertIn("golden1.jpg", df["filename"].values)
            self.assertIn("bulldog1.jpg", df["filename"].values)
            # cette ligne car elle est la source du problème d'ordre
            # self.assertEqual(df.iloc[0]["filename"], "golden1.jpg") 

# ------------------------------------------------------------------------------------------------------------

class TestSearchByFilename(unittest.TestCase):
    def setUp(self):
        """ Crée un dataset simulé avec plusieurs races """
        self.df = pd.DataFrame({
            "filename": ["golden1.jpg", "bulldog1.jpg", "husky1.jpg"],
            "class": ["GoldenRetriever", "Bulldog", "Husky"]
        })

    def test_search_partial(self):
        result = image_search.search_by_filename(self.df, "golden", partial=True)
        self.assertEqual(len(result), 1)  # Doit trouver uniquement le Golden Retriever

    def test_search_exact(self):
        result = image_search.search_by_filename(self.df, "husky1.jpg", partial=False)
        self.assertEqual(len(result), 1)  # Vérifie qu'on trouve exactement 1 Husky

# ------------------------------------------------------------------------------------------------------------

class TestSearchByClass(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "filename": ["golden1.jpg", "bulldog1.jpg", "husky1.jpg"],
            "class": ["GoldenRetriever", "Bulldog", "Husky"]
        })

    def test_search_partial(self):
        result = image_search.search_by_class(self.df, "Bulldog", partial=True)
        self.assertEqual(len(result), 1)  # Vérifie qu’on trouve bien le Bulldog

    def test_search_exact(self):
        result = image_search.search_by_class(self.df, "Husky", partial=False)
        self.assertEqual(len(result), 1)  # Vérifie qu’on trouve exactement 1 Husky

# ------------------------------------------------------------------------------------------------------------

class TestSearchByKeyword(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "filename": ["golden1.jpg", "bulldog1.jpg", "husky1.jpg"],
            "class": ["GoldenRetriever", "Bulldog", "Husky"]
        })

    def test_search_keyword_in_class(self):
        result = image_search.search_by_keyword(self.df, "Husky")
        self.assertEqual(len(result), 1)  # Vérifie qu'on trouve le Husky

    def test_search_keyword_in_filename(self):
        result = image_search.search_by_keyword(self.df, "bulldog")
        self.assertEqual(len(result), 1)  # Vérifie qu'on trouve le Bulldog

# ------------------------------------------------------------------------------------------------------------

class TestShowImages(unittest.TestCase):
    def setUp(self):
        """ Simule des images de différentes races pour les afficher """
        self.test_dir = "data/"
        os.makedirs(self.test_dir, exist_ok=True)

        img1 = Image.new("RGB", (256, 256), (255, 255, 255))  # Fond blanc
        img1.save(os.path.join(self.test_dir, "golden1.jpg"))
        img2 = Image.new("RGB", (256, 256))
        img2.save(os.path.join(self.test_dir, "bulldog1.jpg"))

        self.df = pd.DataFrame({
            "image_path": [os.path.join(self.test_dir, "golden1.jpg"),
                           os.path.join(self.test_dir, "bulldog1.jpg")],
            "class": ["GoldenRetriever", "Bulldog"]
        })

    def tearDown(self):
        os.remove(os.path.join(self.test_dir, "golden1.jpg"))
        os.remove(os.path.join(self.test_dir, "bulldog1.jpg"))
        # Supprimer le dossier "data" seulement s'il est vide
        if not os.listdir(self.test_dir):
            os.rmdir(self.test_dir)

    @patch('matplotlib.pyplot.show') # Simule la fonction plt.show()
    @patch('matplotlib.pyplot.close') # Simule la fonction plt.close() 
    def test_show_images(self, mock_close, mock_show):  # Accept both mock arguments
        """ Vérifie qu'on peut afficher les images sans erreur (sans réellement ouvrir de fenêtre) """
        try:
            image_search.show_images(self.df, max_images=2)
            # Vérifie que plt.show() a été appelé une fois
            mock_show.assert_called_once()
            result = True
        except Exception as e:
            print(f"\nErreur inattendue lors de l'affichage des images: {e}")
            result = False
        self.assertTrue(result)

# ------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()