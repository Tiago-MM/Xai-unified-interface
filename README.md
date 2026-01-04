# 🛡️ Unified Explainable AI Interface

Cette plateforme unifiée permet de détecter des **Deepfakes Audio** (via spectrogrammes) et de diagnostiquer des **Cancers du Poumon** (via radiographies thoraciques) tout en fournissant des explications visuelles grâce à l'IA explicable (XAI).

## 📁 Structure du Projet

- `app.py` : Interface principale Streamlit.
- `models/` : Logique de chargement des architectures CNN (VGG16, ResNet, etc.).
- `explanations/` : Implémentations des méthodes XAI (Grad-CAM, LIME, SHAP).
- `utils/` : Fonctions de prétraitement audio et image.
- `examples/` : Fichiers de test (audio `.wav` et image `.jpg/.png`).
- `requirements.txt` : Liste des dépendances Python.
- `.gitignore` : Fichiers exclus du suivi Git.

## 🚀 Installation

1. **Créer un environnement virtuel :**
   ```bash
   python3 -m venv venv
   source venv/bin/activate

```

2. **Installer les dépendances :**
```bash
pip install -r requirements.txt

```



## 💻 Utilisation

Lancez l'application avec la commande suivante :

```bash
streamlit run app.py

```

### Étapes pour l'analyse :

1. Chargez un fichier depuis le dossier `examples/` ou votre ordinateur.
2. L'interface détecte automatiquement s'il s'agit d'**Audio** ou d'**Image**.
3. Choisissez un modèle et les méthodes XAI (Grad-CAM, LIME, SHAP).
4. Consultez les prédictions et comparez les explications dans l'onglet dédié.
