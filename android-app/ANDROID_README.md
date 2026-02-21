# Deep Live Cam - Android Offline APK

Version Android offline de Deep Live Cam pour le face-swap en temps reel, entierement hors-ligne.

## Architecture

L'application Android utilise :
- **ONNX Runtime Mobile** pour l'inference des modeles IA directement sur l'appareil
- **Google ML Kit** (modele embarque) pour la detection de visages
- **CameraX** pour le mode camera en direct
- **Material Design 3** pour l'interface utilisateur

### Pipeline de traitement

```
1. Detection de visage (ML Kit, offline)
       |
2. Extraction des landmarks (5 points)
       |
3. Alignement du visage (transformation affine)
       |
4. Extraction d'embedding (ArcFace/w600k_r50.onnx)
       |
5. Face Swap (InSwapper/inswapper_128.onnx)
       |
6. Paste-back avec blending (masque elliptique + feathering)
```

## Pre-requis

- Android Studio Hedgehog (2023.1.1) ou plus recent
- Android SDK 34
- JDK 17
- Un appareil Android 8.0+ (API 26+) avec ARM64

## Compilation

### 1. Ouvrir le projet

```bash
cd android-app
```

Ouvrir le dossier `android-app/` dans Android Studio.

### 2. Telecharger les modeles IA

Les modeles IA ne sont PAS inclus dans l'APK en raison de leur taille.
Vous devez les telecharger separement :

| Modele | Taille | Description |
|--------|--------|-------------|
| `inswapper_128.onnx` | ~550 MB | Modele de face swap |
| `w600k_r50.onnx` | ~175 MB | Modele de reconnaissance faciale (ArcFace) |

**Liens de telechargement :**
- inswapper_128.onnx : https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx
- w600k_r50.onnx : https://huggingface.co/hacksider/deep-live-cam/resolve/main/w600k_r50.onnx

### 3. Placer les modeles

**Option A** - Embarquer dans l'APK (APK sera ~750MB) :
```
android-app/app/src/main/assets/models/inswapper_128.onnx
android-app/app/src/main/assets/models/w600k_r50.onnx
```

**Option B** - Copier sur l'appareil apres installation :
```
Android/data/com.deeplivecam/files/models/inswapper_128.onnx
Android/data/com.deeplivecam/files/models/w600k_r50.onnx
```

Utiliser `adb push` :
```bash
adb push inswapper_128.onnx /storage/emulated/0/Android/data/com.deeplivecam/files/models/
adb push w600k_r50.onnx /storage/emulated/0/Android/data/com.deeplivecam/files/models/
```

### 4. Compiler l'APK

```bash
# Debug APK
./gradlew assembleDebug

# Release APK (necessite une cle de signature)
./gradlew assembleRelease
```

L'APK genere se trouve dans :
```
app/build/outputs/apk/debug/app-debug.apk
app/build/outputs/apk/release/app-release.apk
```

### 5. Installer

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Utilisation

### Mode Image (Face Swap statique)

1. Ouvrir l'application
2. Toucher "Source Face" pour selectionner le visage source
3. Toucher "Target Image" pour selectionner l'image cible
4. Appuyer sur "Start Face Swap"
5. Sauvegarder le resultat dans la galerie

### Mode Camera (Face Swap en direct)

1. Selectionner d'abord un visage source
2. Appuyer sur "Live Camera"
3. Le face swap s'applique en temps reel sur le flux camera
4. Utiliser le bouton capture pour prendre une photo

### Mode Part Swap (N'importe quelle region)

Ce mode permet de coller n'importe quelle image sur n'importe quelle zone -
pas seulement les visages. Aucun modele IA requis pour ce mode.

**Cas d'utilisation :**
- Appliquer un tatouage sur la peau
- Coller un logo sur un vetement
- Remplacer une partie du corps
- Superposer une texture sur une surface
- Mode AR/sticker en temps reel

**Utilisation :**
1. Depuis l'ecran principal, toucher "Part Swap (Any Region)"
2. Selectionner l'image source (la "part" a coller)
3. Selectionner l'image cible
4. Glisser les 4 points de controle pour positionner la zone
5. Ajuster l'opacite et le feathering des bords
6. Appuyer sur "Apply" pour le resultat final
7. Sauvegarder dans la galerie

**Mode camera live :**
- Les 4 points de controle sont draggables en direct
- L'image source est warped en perspective sur le flux camera
- Ajuster l'opacite en temps reel

## Structure du projet

```
android-app/
├── app/
│   ├── build.gradle.kts          # Dependencies et config
│   ├── proguard-rules.pro        # Regles ProGuard
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/models/         # Modeles IA (a telecharger)
│       ├── java/com/deeplivecam/
│       │   ├── DeepLiveCamApp.kt         # Application class
│       │   ├── ml/
│       │   │   ├── FaceDetector.kt       # Detection (ML Kit)
│       │   │   ├── FaceRecognizer.kt     # Embedding (ArcFace ONNX)
│       │   │   ├── FaceSwapper.kt        # Face swap (InSwapper ONNX)
│       │   │   └── ModelManager.kt       # Gestion des modeles
│       │   ├── processing/
│       │   │   ├── FaceAligner.kt        # Alignement + paste-back
│       │   │   ├── FaceSwapPipeline.kt   # Pipeline face swap
│       │   │   ├── PartSwapPipeline.kt   # Pipeline part swap
│       │   │   └── RegionWarper.kt       # Perspective warp + blend
│       │   ├── ui/
│       │   │   ├── MainActivity.kt       # Ecran principal
│       │   │   ├── LiveCameraActivity.kt # Mode camera face swap
│       │   │   ├── PartSwapActivity.kt   # Mode part swap image
│       │   │   ├── PartSwapLiveActivity.kt # Mode part swap camera
│       │   │   ├── TouchPointOverlay.kt  # 4 points de controle draggables
│       │   │   └── ProcessingActivity.kt # Traitement batch
│       │   └── utils/
│       │       └── BitmapUtils.kt        # Utilitaires image
│       └── res/
│           ├── layout/                   # Layouts XML
│           ├── values/                   # Strings, couleurs, themes
│           └── drawable/                 # Drawables
├── build.gradle.kts              # Config projet racine
├── settings.gradle.kts
└── gradle/
```

## Performances

| Appareil | Face Swap (image) | Live FPS |
|----------|-------------------|----------|
| Snapdragon 8 Gen 2+ | ~1-2s | ~5-10 FPS |
| Snapdragon 7xx | ~3-5s | ~2-5 FPS |
| Appareils entree de gamme | ~5-10s | Non recommande |

Les performances dependent fortement du processeur de l'appareil.
Le mode live est optimise pour les appareils haut de gamme.

## Limitations par rapport a la version desktop

- **Pas de GFPGAN** : L'amelioration de visage n'est pas disponible (trop lourd pour mobile)
- **Pas de traitement video** : Uniquement images et camera en direct
- **Resolution limitee** : Les images sont redimensionnees pour les performances
- **Pas de Poisson blending** : Utilise un blending elliptique avec feathering
- **Pas de mouth mask** : Pas de restauration de la bouche originale

## Licence

Meme licence que le projet Deep Live Cam principal.
Usage responsable et ethique uniquement.
