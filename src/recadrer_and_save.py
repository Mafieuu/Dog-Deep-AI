import os
import xml.etree.ElementTree as ET
from PIL import Image

""" Exemple d'annotation
<annotation>
	<folder>02085782</folder>
	<filename>n02085782_17</filename>
	<source>
		<database>ImageNet database</database>
	</source>
	<size>
		<width>393</width>
		<height>425</height>
		<depth>3</depth>
	</size>
	<segment>0</segment>
	<object>
		<name>Japanese_spaniel</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>48</xmin>
			<ymin>102</ymin>
			<xmax>367</xmax>
			<ymax>367</ymax>
		</bndbox>
	</object>
</annotation>
"""
def recadrer_save(img_path,annot_path,dest_dir):
    """
    Permet de recadrer une image compte tenu de la boite xml du/des chiens
    """

    img = Image.open(img_path)
    tree = ET.parse(annot_path)
    root = tree.getroot() # le root du fichier xml
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin, ymin, xmax, ymax = [int(bbox.find(k).text) for k in ('xmin','ymin','xmax','ymax')]
        img_extrait = img.crop((xmin, ymin, xmax, ymax))
        fn = os.path.basename(img_path)
        img_extrait.save(os.path.join(dest_dir, fn))
