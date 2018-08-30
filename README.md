# Sparsize


A partir d'une image neural_sparse tentent de créer une nouvelle image proche mais avec un encodage à une certaine couche d'un réseau de neurone plus parcimonieux.
(ici VGG19). Pour faire cela l'algorithme tente de trouver l'argument minimum de cette quantité:

![Alt](/CodeCogsEqn.gif "erreur 1")

neural_sparseHALF ne contraint pas seulement l'encodage à la couche k mais la moitié supérieur des couches de convolutions du réseau.

neural_sparseALL contraint toutes les couches de convolution du réseau.

Pour utiliser neural_sparse taper dans l'invite commande:

python neural_sparse.py --img example.jpg --layer relu1-1 --output out.jpg

Pour réécrire la sortie sur un fichier existant, il faut y ajouter --overwrite.
Pour changer le \lambda il suffit d'y ajouter --regularisation_coeff 3. (Attention vous devez bien préciser que cette variable est un float)

Pour neural_sparseHALF et ALL il n'y a pas besoin d'indiquer la couche "layer".

Bibliothèques utilisées:
- Pillow
- Scipy
- Tensorflow
- Numpy

Largment inspiré par l'implémentation du transfert de style par Anish Athalye : https://github.com/anishathalye/neural-style
