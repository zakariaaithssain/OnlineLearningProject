# Mini-projet CNN - Integration de TP1 a TP5

## 1. Vue d'ensemble

Ce depot contient un mini-projet de reseaux de neurones convolutionnels (CNN)
concu pour integrer, dans un seul cadre experimental, les notions vues dans
TP1 a TP5.

Le projet s'appuie sur un jeu de donnees d'images de type CelebA et doit
couvrir deux taches:

- une tache de regression
- une tache de classification binaire

Le cadre experimental retenu actuellement est le suivant:

- **Dataset vise**: CelebA reduced
- **CNN1**: architecture simple
- **CNN2**: architecture amelioree
- **Regression**: prediction de la somme de tous les attributs disponibles
- **Classification binaire**: prediction de l'attribut `Smiling`
- **Formalisme classification aligne avec le PDF**:
  - labels en `{-1, +1}`
  - score brut du reseau
  - decision par `sign(score)`
  - perte de type hinge

Le document de reference du mini-projet est:

- `mini_projet.pdf`

Guide complementaire utile:

- [training_reuse_guide.md](./training_reuse_guide.md) pour savoir quand reutiliser un entrainement et quand re-entrainer selon TP1 a TP5

## 2. Objectif pedagogique

Le but n'est pas seulement d'entrainer un modele, mais de relier les concepts
theoriques des TPs a des experiences concretes sur deux architectures CNN
differentes.

Le projet final doit permettre de:

- comparer deux architectures CNN
- traiter une regression et une classification binaire
- analyser la complexite des donnees
- etudier gradient, sous-gradient, line search, regularisation
- comparer plusieurs optimiseurs
- introduire des variantes online et stochastiques
- produire des courbes, tableaux et conclusions argumentees

## 3. Structure actuelle du depot

Le projet conserve volontairement une structure simple, avec les fichiers
principaux a la racine.

```text
.
├── README.md
├── mini_projet.pdf
├── pyproject.toml
├── uv.lock
├── .gitignore
├── main.py
├── experiment_spec.py
├── data_loader.py
├── train_common.py
├── train_classification.py
├── train_regression.py
├── utils.py
├── tp1_related.ipynb
└── models/
    ├── __init__.py
    ├── cnn_simple.py
    └── cnn_improved.py
```

### Role de chaque fichier

#### `README.md`

Document de presentation du projet:

- objectif
- structure du repo
- commandes principales
- avancement

#### `mini_projet.pdf`

Sujet officiel du mini-projet. C'est la source de reference pour verifier que
le travail est bien aligne avec les consignes.

#### `pyproject.toml`

Configuration du projet Python et declaration des dependances:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- etc.

#### `uv.lock`

Fichier de lock des dependances pour reproduire l'environnement avec `uv`.

#### `.gitignore`

Ignore notamment:

- `data/`
- les PDFs temporaires
- `__pycache__/`

#### `main.py`

Point d'entree principal du projet.

Il sert a:

- afficher l'etat du projet avec `status`
- lancer la classification avec `train-classification`
- lancer la regression avec `train-regression`

#### `experiment_spec.py`

Centralise le cadre experimental retenu.

Ce fichier fixe:

- le dataset cible
- les deux modeles utilises
- la tache de regression
- la tache de classification
- la convention de labels
- la perte de classification

Son role est d'eviter de disperser les choix experimentaux dans plusieurs
scripts.

#### `data_loader.py`

Gestion du chargement des donnees.

Ce fichier contient:

- la classe `CelebADataset`
- la construction des transformations image
- la gestion des splits `train/val/test`
- la creation des `DataLoader`
- la generation de la cible de regression
- la conversion des labels de classification selon le schema choisi

#### `train_common.py`

Briques communes pour l'entrainement.

On y trouve:

- la creation des modeles a partir du registre
- la perte hinge binaire
- la selection du device
- la boucle d'entrainement d'une epoch
- l'evaluation pour la regression
- l'evaluation pour la classification
- quelques utilitaires de parsing et de sauvegarde

#### `train_classification.py`

Script CLI dedie a la classification binaire.

Il gere:

- les arguments de ligne de commande
- la creation des DataLoaders
- la selection du modele
- la perte hinge
- l'entrainement
- la validation
- la sauvegarde du meilleur checkpoint
- l'export des metriques

#### `train_regression.py`

Script CLI dedie a la regression.

Il gere:

- les arguments de ligne de commande
- la definition de la cible de regression
- l'entrainement en MSE
- la validation
- l'evaluation finale
- l'export des metriques

#### `utils.py`

Fichier de fonctions utilitaires liees aux TPs.

Il contient deja des briques pour:

- covering number
- line search
- validation croisee
- regularisation L1/L2
- metriques de classification
- projection sur boule L2
- normalisation
- plotting
- normes et normes duales
- regret

Ce fichier sert de base theorique et algorithmique pour relier le projet aux
consignes TP1 a TP5.

#### `tp1_related.ipynb`

Notebook dedie a TP1.

Il sert a:

- explorer le dataset
- construire ou verifier le split
- analyser les attributs
- etudier le covering number
- ajouter progressivement les experiences TP1

Il a ete enrichi pour couvrir davantage les attentes du sujet TP1.

#### `models/__init__.py`

Point d'entree du package de modeles. Pour l'instant, il est minimal.

#### `models/cnn_simple.py`

Implementation de `CNN1`, une architecture volontairement simple.

Caracteristiques:

- peu de couches convolutionnelles
- architecture legere
- sert de baseline

#### `models/cnn_improved.py`

Implementation de `CNN2`, une architecture plus riche.

Caracteristiques:

- plus profonde
- batch normalization
- dropout
- capacite plus elevee

Elle sert a comparer la complexite du modele et l'impact sur biais/variance et
performance.

## 4. Donnees attendues

Le repo ne versionne pas les donnees. Le code suppose que le dossier suivant
existe localement:

```text
data/
├── img_align_celeba_reduced/
├── list_attr_celeba_reduced.csv
├── list_attr_celeba.csv
├── identity_celeba_reduced.csv
└── list_eval_partition_reduced.csv
```

Selon les scripts, le projet peut aussi retomber sur:

- `data/img_align_celeba`
- `data/list_attr_celeba.csv`
- `data/list_eval_partition.csv`

Sans le dossier `data/`, les scripts d'entrainement et le notebook ne peuvent
pas aller jusqu'au bout.

## 5. Commandes utiles

### Installation de l'environnement

Avec `uv`:

```bash
uv sync
```

Si besoin, une version CPU de PyTorch peut etre installee via:

```bash
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch torchvision
```

### Voir l'etat du projet

```bash
python3 main.py status
```

ou avec l'environnement du projet:

```bash
.venv/bin/python main.py status
```

### Lancer la classification

```bash
.venv/bin/python train_classification.py
```

### Lancer la regression

```bash
.venv/bin/python train_regression.py
```

## 6. Avancement actuel

### Ce qui est deja fait

- Le sujet du mini-projet a ete lu et compare au code existant.
- Un cadre experimental explicite a ete fixe dans `experiment_spec.py`.
- Deux architectures CNN sont en place:
  - `CNN1` simple
  - `CNN2` improved
- Les scripts d'entrainement existent pour:
  - la classification
  - la regression
- La classification a ete recalee sur le formalisme du PDF:
  - score brut
  - labels signes `{-1, +1}`
  - perte hinge
- Le notebook `tp1_related.ipynb` a ete complete pour mieux couvrir TP1:
  - exploration
  - covering number
  - bruit / valeurs atypiques / densite
  - gradient differentiable
  - line search
  - biais-variance
  - validation et regularisation
- `torch` et les dependances principales ont ete installes dans la `.venv`.

### Ce qui bloque encore

- Le dossier `data/` est absent du repo local au moment de la redaction de ce
  README.
- A cause de cela, les cellules d'entrainement du notebook et les scripts ne
  peuvent pas etre executes jusqu'au bout.

### Ce qui reste a faire

- Ajouter les donnees dans `data/`
- Executer proprement le notebook TP1
- Verifier et corriger les cellules si des erreurs runtime apparaissent
- Creer un notebook dedie pour chaque TP:
  - `tp2_related.ipynb`
  - `tp3_related.ipynb`
  - `tp4_related.ipynb`
  - `tp5_related.ipynb`
- Ajouter les experiences comparees demandees par le sujet
- Produire les tableaux, figures et conclusions finales
- Rediger le rapport final

## 7. Proposition de progression

Ordre conseille pour la suite:

1. remettre les donnees dans `data/`
2. executer completement `tp1_related.ipynb`
3. verifier que `train_classification.py` et `train_regression.py` tournent
4. creer un notebook par TP en gardant la meme structure simple
5. centraliser les resultats dans `outputs/` si besoin
6. ecrire le rapport final a partir des notebooks

## 8. Idee directrice du projet

Le projet doit rester lisible et pedagogique.

La priorite n'est pas de construire une grosse architecture logicielle, mais de
garder un depot simple ou l'on comprend rapidement:

- quel fichier fait quoi
- quelle experience correspond a quel TP
- quelles hypotheses ont ete choisies
- ou trouver les resultats

Dans cette logique, garder une structure simple avec:

- un script de regression
- un script de classification
- un notebook par TP
- un fichier de specification experimentale

est une bonne approche pour finir proprement le mini-projet.
