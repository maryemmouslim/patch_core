# LightShift-PatchCore

PatchCore pour la détection d'anomalies industrielles MVTec AD, avec quatre variantes :

- baseline PatchCore ;
- B1 : banque mémoire enrichie par variations photométriques ;
- B2 : alignement statistique des caractéristiques ;
- B3 : seuil conforme sur des images normales sous l'éclairage cible.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Dataset

Le dossier `mvtec_anomaly_detection` doit contenir les catégories MVTec directement à sa racine.

## Baseline sur bottle

```powershell
python -m src.experiments.run_experiments --root mvtec_anomaly_detection --categories bottle --lighting reference
```

Résultats : `artifacts/bottle/reference/metrics.json` et `artifacts/results.csv`.

## Changements d'éclairage

```powershell
python -m src.experiments.run_experiments --root mvtec_anomaly_detection --categories bottle --lighting dim
python -m src.experiments.run_experiments --root mvtec_anomaly_detection --categories bottle --lighting bright
```

Variantes proposées :

```powershell
python -m src.experiments.run_experiments --categories bottle --lighting dim --b1
python -m src.experiments.run_experiments --categories bottle --lighting dim --b2
python -m src.experiments.run_experiments --categories bottle --lighting dim --b3
python -m src.experiments.run_experiments --categories bottle --lighting dim --b1 --b2 --b3
```

Toutes les catégories :

```powershell
python -m src.experiments.run_experiments --categories bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper --lighting dim --b1 --b2 --b3
```

## Tests

```powershell
python -m pytest tests
```

## Notebooks

Installer les dépendances puis ouvrir VS Code/Jupyter depuis la racine du projet :

```powershell
python -m pip install -r requirements.txt
python -m ipykernel install --user --name lightshift --display-name "Python (LightShift)"
code .
```

Dans VS Code, sélectionner le kernel **Python (LightShift)** pour les fichiers de `notebooks/`.
