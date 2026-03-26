# Reutilisation de l'entrainement des CNN

## Idee generale

**Oui, on peut entrainer les deux CNN une premiere fois et reutiliser ces modeles ensuite, mais seulement comme baseline.**

Le bon reflexe est:

1. entrainer `CNN1` et `CNN2` sur les deux taches principales
2. sauvegarder checkpoints, metriques et courbes
3. reutiliser ces resultats comme reference dans les notebooks suivants
4. re-entrainer uniquement quand un TP change la methode d'apprentissage

Donc, le premier entrainement n'est pas perdu: il sert de **point de comparaison commun** pour tous les TPs.

## Ce qu'il faut entrainer au minimum au debut

Le bloc baseline ideal contient:

- `CNN1` pour la regression
- `CNN2` pour la regression
- `CNN1` pour la classification binaire
- `CNN2` pour la classification binaire

Ces 4 entrainements servent ensuite de base pour:

- comparer les architectures
- analyser biais / variance
- mesurer l'effet d'une nouvelle loss
- mesurer l'effet d'un nouvel optimiseur
- juger si une methode online ou stochastique apporte un gain

## Quand la reutilisation est valide

Tu peux reutiliser un modele deja entraine si le notebook:

- commente des resultats deja obtenus
- compare `CNN1` et `CNN2`
- etudie les performances finales
- montre des courbes deja calculees
- discute underfitting / overfitting sur un protocole identique
- reutilise exactement la meme loss, le meme optimiseur, et le meme cadre d'entrainement

Autrement dit:

- si le notebook **analyse**, tu peux reutiliser
- si le notebook **change la methode d'apprentissage**, tu dois re-entrainer

## Quand il faut absolument re-entrainer

Il faut re-entrainer quand un TP introduit un changement sur:

- la fonction de cout
- l'optimiseur
- la direction de mise a jour
- la regularisation
- le mode d'apprentissage online
- le mode stochastique
- le protocole experimental

Exemples typiques:

- passer de MSE a une autre loss
- passer de BCE/hinge a sous-gradient explicite
- comparer SGD, Adam, RMSProp, Momentum
- passer a l'apprentissage online
- tester une regularisation absente du baseline

## Application detaillee sur les 5 TPs

### TP1

TP1 est le **meilleur endroit pour faire les entrainements de base**.

Pourquoi:

- on y pose le cadre experimental
- on y compare `CNN1` et `CNN2`
- on y discute biais / variance
- on y met en place train / validation / test
- on y introduit la regularisation

Donc pour TP1:

- **faire les 4 entrainements baseline est recommande**
- ces resultats peuvent ensuite etre reutilises dans les autres notebooks

Ce qu'on peut reutiliser plus tard depuis TP1:

- checkpoints des deux CNN
- metriques finales train/val/test
- courbes de loss
- comparaison `CNN1` vs `CNN2`
- observations sur underfitting / overfitting

### TP2

TP2 travaille la **classification binaire en cadre non differentiable**.

Ici, le point crucial est que la methode change:

- sortie scalaire brute
- regle `sign(score)`
- perte hinge
- sous-gradient

Donc:

- si ton baseline TP1 utilisait deja **exactement** ce cadre hinge pour la classification, tu peux reutiliser une partie des resultats
- mais pour le vrai TP2, il faut en pratique **re-entrainer au moins la classification** afin de montrer:
  - la perte non differentiable
  - le sous-gradient
  - le choix de direction
  - la validation / regularisation dans ce nouveau cadre

Conclusion TP2:

- **reutilisation partielle possible**
- **re-entrainement recommande pour la classification**

### TP3

TP3 porte sur la **comparaison des optimiseurs**:

- gradient standard
- Momentum
- Nesterov
- AdaGrad
- RMSProp
- Adam

Comme l'optimiseur change, il faut re-entrainer.

Ce qu'on reutilise de TP1 / TP2:

- le dataset
- les splits
- les deux architectures
- la tache de regression
- la tache de classification

Ce qu'on doit refaire:

- les entrainements avec chaque optimiseur

Conclusion TP3:

- **re-entrainement obligatoire**

### TP4

TP4 introduit:

- gradient online
- sous-gradient online
- gradient stochastique
- sous-gradient stochastique

Ici encore, le mode d'apprentissage change profondement.

Tu ne peux donc pas te contenter du baseline TP1.

Ce qu'on reutilise:

- les architectures
- les taches
- les donnees
- les metriques baseline comme point de comparaison

Ce qu'on doit refaire:

- les mises a jour online / stochastiques
- les mesures de regret ou pertes cumulatives

Conclusion TP4:

- **re-entrainement obligatoire**

### TP5

TP5 va encore plus loin:

- apprentissage supervise en ligne
- algorithmes du premier ordre
- prediction with expert advice
- regularisation online
- noyaux
- normes et normes duales

Ce TP ne consiste pas seulement a reprendre les modeles de base, mais a
introduire de nouvelles methodes d'apprentissage ou de nouvelles variantes.

Donc:

- les modeles baseline restent utiles comme reference
- mais les experiences TP5 doivent etre refaites avec les methodes du TP

Conclusion TP5:

- **re-entrainement obligatoire**

## Resume tres simple par TP

- **TP1**: entrainer les deux CNN une premiere fois, garder les resultats comme baseline
- **TP2**: reutiliser l'idee generale, mais re-entrainer si tu passes au cadre hinge / sous-gradient
- **TP3**: re-entrainer pour chaque optimiseur
- **TP4**: re-entrainer pour les variantes online / stochastiques
- **TP5**: re-entrainer pour les methodes online, experts, noyaux, regularisation online

## Regle pratique finale

Tu peux reutiliser un entrainement seulement si:

- la tache ne change pas
- la loss ne change pas
- l'optimiseur ne change pas
- le mode d'apprentissage ne change pas
- le protocole de validation ne change pas

Si un de ces blocs change, considere qu'il faut **faire une nouvelle experience**.

## Bonne pratique

Toujours sauvegarder:

- les checkpoints des modeles
- les metriques
- les courbes principales
- le nom du CNN utilise
- la tache concernee
- la loss
- l'optimiseur

Ainsi:

- TP1 fournit la base
- TP2 a TP5 reutilisent la base comme reference
- mais chaque TP peut produire ses propres experiences sans confusion
