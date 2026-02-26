# model_adapt_poison

## Description
Ce projet contient des scripts Python pour entraîner et adapter des modèles EfficientNet, avec gestion de modèles propres et empoisonnés.

## Structure du projet
- app.py : Script principal pour l'adaptation et l'entraînement des modèles.
- projet_Arphanipynb.ipynb : Notebook pour l'expérimentation et la visualisation.
- saved_models/ : Dossier contenant les modèles sauvegardés (efficientnet_clean_final.keras, efficientnet_poisoned_final.keras).
- env/ : Environnement virtuel Python.
- .gitignore : Fichier pour exclure les fichiers/dossiers inutiles du contrôle de version.

## Installation
1. Créez un environnement virtuel :
   ```bash
   python -m venv env
   ```
2. Activez l'environnement :
   - Windows : env\Scripts\activate
   - Mac/Linux : source env/bin/activate
3. Installez les dépendances nécessaires (ajoutez-les dans requirements.txt si besoin).

## Utilisation
Lancez le script principal :
```bash
python app.py
```

Ou ouvrez le notebook pour explorer les modèles.

## Auteur
Arphan
