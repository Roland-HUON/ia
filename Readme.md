# Deep Learning - Classification d'images CIFAR-10

Projet de reconnaissance d'objets utilisant le Transfer Learning avec des modèles pré-entraînés (AlexNet et ResNet-18) sur le dataset CIFAR-10.

**Accéder au github pages** : [roland-huon.github.io/ia](https://roland-huon.github.io/ia/)

## Description

Application web de classification d'images capable de reconnaître 10 catégories d'objets :
✈️ Avion • 🚗 Automobile • 🐦 Oiseau • 🐱 Chat • 🦌 Cerf • 🐕 Chien • 🐸 Grenouille • 🐴 Cheval • 🚢 Bateau • 🚚 Camion

**Technologies** :
- PyTorch (Transfer Learning depuis ImageNet)
- ONNX Runtime Web (inférence dans le navigateur)
- Dataset CIFAR-10 (images 32×32 redimensionnées en 224×224)

Toutes les demandes ont été réalisées ainsi que tous les bonus disponibles.

## Membre du projet
- [Roland HUON](https://github.com/Roland-HUON)

## Test entre AlexNet et ResNet-18

### AlexNet VS ResNet-18 : 
Test réalisé sur epoch = 10
- AlexNet : 89,5% max Accuracy | Best Avg loss: 0.308420 | Temps : 23m 45s
- ResNet-18 : 94,3% max Accuracy | Best Avg loss: 0.172854 | Temps : 26m 54s

Ceci s'explique par les caractéristiques de AlexNet (CNN de 8 couches) et de ResNet-18 ( CNN de 18 couches).

Vous pouvez voir la comparaison en cliquant sur ce lien de la ResearchGate : [cliquez](https://www.researchgate.net/figure/Comparison-of-AlexNet-and-ResNet-18_fig4_343955468)

## Images TensorBoard
![TensorBoard](tensorboard_screen.png)
![TensorBoard_Final](tensorboard_screen_final.png)

## Utilisation

1. **Entraînement** : Ouvrir `temp.ipynb` et exécuter "Run All"
2. **Frontend** : Ouvrir `index.html` dans un navigateur (via Laragon ou serveur HTTP)
3. **Prédiction** : Charger une image ou utiliser votre caméra et cliquer sur "Analyser l'image"