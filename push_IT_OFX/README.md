# push_IT_OFX — plug-in OFX pour Nuke (Windows, CPU + CUDA)

Edge extend / propagation de pixels pour Nuke.

- **Expand** : prolonge les couleurs du matte vers l'extérieur (edge extend avant un keyer, un despill, un merge sur un nouveau fond…).
- **Contract** : remplit la zone du matte avec les pixels qui l'entourent (clean plate rapide, suppression d'objet).

## Entrées

| Entrée  | Rôle |
|---------|------|
| Source  | Image à traiter |
| Matte   | (optionnelle) Matte définissant la zone à conserver. Sans elle, l'alpha de Source est utilisé |
| Vector  | (optionnelle) Vecteurs de direction pour le mode Vector Warp (R = X, G = Y, en pixels) |

## Paramètres

**Spread**
- *Spread Type*
  - **Interpolate** : remplissage lisse par pyramide push-pull (équivalent d'un edge extend « propre »).
  - **Stretch** : chaque pixel prend la couleur du pixel de bord le plus proche (transformée de distance exacte) — garde le grain et le contraste du bord.
  - **Vector Warp** : étirement directionnel, le long de l'entrée Vector ou d'un angle constant. Les pixels qui ne trouvent rien dans la direction sont complétés par Interpolate.
- *Spread Mode* : Expand / Contract.
- *Max Distance* / *Distance Falloff* : limite la propagation à N pixels, avec fin douce (0 = illimitée).
- *Spread Blur* : flou appliqué uniquement aux pixels propagés.
- *Seed Threshold* : valeur de matte à partir de laquelle un pixel peut être copié (Stretch, Max Distance).

**Matte**
- *Matte Source* : Auto, Source Alpha, Matte Alpha, Matte Luminance.
- *Invert Matte*.
- *Source Premultiplied* : en Expand, dépremultiplie la source pour que les bords ne noircissent pas.
- *Edge Shrink* : érode la zone conservée de N pixels pour ignorer les bords contaminés (très utile en Contract).
- *Edge Softness* : adoucit la transition entre pixels d'origine et pixels propagés.

**Vector Warp**
- *Vector Source* : Vector Input ou Constant.
- *Angle* / *Length* (Constant) : direction (0° = vers la droite, 90° = vers le haut) et longueur de l'étirement.
- *Motion Distance* : gain sur les vecteurs de l'entrée (négatif = sens inverse).
- *Motion Threshold* : ignore les vecteurs plus courts que ce seuil.
- *Vector Origin* et *Red/Green/Blue X/Y Gain* : matrice qui transforme les canaux RGB de l'entrée en déplacement X/Y.
- *Invert Direction*.

**Output**
- *Result Output* : Combined (original + propagation, RGB non premultiplié), Spread Only, ou **Premult** (Combined premultiplié par l'alpha choisi dans *Alpha Output*).
- *Alpha Output* : Input, Spread Matte (matte étendu), Solid.
- *Extend BBox to Format* : agrandit la bbox au format pour propager jusqu'aux bords de l'image.

**GPU** (première ligne du panneau)
- *Use GPU (CUDA)* : active le calcul sur la carte NVIDIA. Si CUDA est absent ou échoue, le rendu repasse automatiquement sur le CPU (multithreadé).
- À droite de la case : la carte détectée.

> Nuke ne fournit pas de buffers GPU aux plug-ins OFX : le plug-in envoie l'image sur le GPU, calcule tout le traitement, puis la récupère. Le gain est net dès que l'image est grande ou que Vector Warp / Spread Blur sont utilisés avec de fortes valeurs.

## Performances

- Pixel source le plus proche (Stretch, Vector Warp, Max Distance) : transformée de distance exacte de Felzenszwalb, en 2 passes sur l'image.
- Flous (Spread Blur, Edge Softness) : 3 passes de boîte glissante par direction, calculées ligne par ligne et par blocs de colonnes qui restent dans le cache ; coût indépendant de la taille du flou, sur CPU comme sur GPU.
- Sans Spread Blur, le remplissage et le composite final sont faits en un seul passage.
- Mesure de référence (HD, 6 modes de test, CPU 2 cœurs) : 0,51 s au total, contre 1,29 s pour la version précédente.

## Compilation (Windows)

Prérequis :
- Visual Studio 2022 (charge de travail « Développement Desktop en C++ »)
- CMake ≥ 3.20 et Git (le SDK OpenFX est téléchargé automatiquement depuis GitHub)
- Optionnel : **CUDA Toolkit 12.x ou 13.x** pour l'accélération GPU

```bat
build.bat          :: CPU + CUDA (si le CUDA Toolkit est installé)
build.bat cpu      :: CPU seulement
```

Ou à la main :

```bat
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Options CMake utiles :
- `-DPUSHIT_WITH_CUDA=OFF` : pas de CUDA.
- `-DPUSHIT_CUDA_ARCHS="75-real;86-real"` : architectures GPU ciblées. Par défaut : Turing (RTX 20xx), Ampere (RTX 30xx / A-series), Ada (RTX 40xx), et avec CUDA 12.8+ Blackwell (RTX 50xx), + PTX compilé à la volée pour les cartes plus récentes. Pour une Pascal (GTX 10xx), utilisez CUDA 12 et ajoutez `61-real` (CUDA 13 ne la gère plus).
- `-DOPENFX_ROOT=C:/dev/openfx` : utiliser un SDK OpenFX local au lieu de le télécharger.

Le runtime CUDA est lié statiquement : aucune DLL à copier, il suffit d'un pilote NVIDIA récent.

## Compilation et installation (Linux Mint 22 / Ubuntu 24.04)

Version précompilée : extrayez `push_IT_OFX-linux-x86_64.tar.gz` puis
```bash
sudo mkdir -p /usr/OFX/Plugins
sudo cp -r push_IT_OFX.ofx.bundle /usr/OFX/Plugins/
```
(ou placez le bundle ailleurs et ajoutez ce dossier à `OFX_PLUGIN_PATH`).

Pour compiler soi-même :
```bash
sudo apt install build-essential cmake git
./build.sh            # CPU + CUDA si nvcc est trouvé
./build.sh install    # copie dans /usr/OFX/Plugins (sudo)
```
Pour le GPU : pilote NVIDIA 580 ou plus récent (Gestionnaire de pilotes de Mint) et, pour compiler, le CUDA Toolkit 13 depuis le dépôt NVIDIA pour Ubuntu 24.04. Le runtime C++ et le runtime CUDA sont intégrés au plug-in : il ne dépend que de la glibc (2.38 minimum).

Le script de couleur du nœud s'installe de la même façon que sous Windows, dans `~/.nuke/`.

## Installation dans Nuke

Copiez le dossier `build\push_IT_OFX.ofx.bundle` dans :

```
C:\Program Files\Common Files\OFX\Plugins\
```

ou placez-le dans un dossier de votre choix et ajoutez ce dossier à la variable d'environnement `OFX_PLUGIN_PATH`.

### Couleur du nœud dans le Node Graph

OFX ne permet pas à un plug-in de choisir sa couleur dans Nuke : elle est appliquée par un petit script Python.

1. Copiez `nuke\push_it_nuke.py` dans `%USERPROFILE%\.nuke\`
2. Ajoutez cette ligne dans `%USERPROFILE%\.nuke\menu.py` (créez le fichier s'il n'existe pas) :
   ```python
   import push_it_nuke
   ```

Couleur par défaut : R 0.7, G 0.592, B 0.21 (modifiable en haut du script). Une couleur changée à la main sur un nœud est conservée.

Au lancement de Nuke, le nœud apparaît sous **Filter > push_IT_OFX** (ou via Tab, nom « push_IT_OFX »). Si le nœud n'apparaît pas, supprimez le cache OFX de Nuke (`%USERPROFILE%\.nuke\ofxplugincache` ou le dossier `ofxplugincache` dans le répertoire temporaire de Nuke) et relancez.

## Recettes

- **Edge extend avant despill/merge** : Source = plate prémultipliée, Expand, Interpolate, Edge Shrink 1–2, Alpha Output = Input. Premultipliez ensuite.
- **Clean plate rapide** : Source = plate, Matte = roto de l'objet, Contract, Interpolate ou Stretch, Edge Shrink 2–5, Spread Blur selon le grain.
- **Étirement directionnel** : Vector Warp, Vector Source = Constant, réglez l'angle dans la direction où les pixels doivent couler. Pour un contrôle local, branchez une passe de vecteurs (motion vectors en pixels, ou une UV « déplacée ») dans l'entrée Vector.

## Structure du code

```
src/push_IT_OFXCommon.h    maths par pixel partagées CPU / GPU (__host__ __device__)
src/push_IT_OFXCPU.cpp     pipeline CPU multithreadé
src/push_IT_OFXCUDA.cu     pipeline CUDA (mêmes étapes, buffers GPU mis en cache)
src/push_IT_OFXCUDAStub.cpp version sans CUDA
src/push_IT_OFXPlugin.cpp  plug-in OFX (paramètres, clips, RoD/RoI, rendu)
```

Pipeline : matte → zone conservée (érosion, adoucissement) → couleurs pondérées → transformée de distance exacte (pixel le plus proche + distance) → pyramide push-pull → remplissage selon le type (+ marche le long du vecteur accélérée par le champ de distance) → flou de propagation → composite.
