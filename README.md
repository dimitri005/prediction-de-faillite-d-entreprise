# Prediction de Faillite d'Entreprise

Pipeline de machine learning pour la prediction de faillite d'entreprises a partir de ratios financiers — base sur le Polish Companies Bankruptcy Dataset (UCI).

## Contexte

La faillite d'une entreprise est rarement un evenement soudain. Les signaux financiers precedent la rupture de plusieurs exercices. Ce projet construit un systeme de detection precoce exploitant des indicateurs comptables et financiers pour classifier le risque de defaillance.

## Dataset

- **Source** : Polish Companies Bankruptcy Dataset — UCI Machine Learning Repository
- **Periode** : donnees financieres sur 5 ans
- **Volume** : plusieurs milliers d'entreprises, classes binaires (faillite / non-faillite)
- **Desequilibre** : forte predominance de la classe saine — traite via StratifiedKFold et poids de classe

## Pipeline ML

### Feature Engineering
- Renommage des ratios financiers en francais (lisibilite metier)
- Traitement des valeurs manquantes et outliers
- Selection de variables par RFE (Recursive Feature Elimination) via LightGBM

### Modelisation
- **Algorithme retenu** : LightGBM (gradient boosting sur arbres)
- **Validation** : StratifiedKFold 5 splits
- **Comparaison** : 5 algorithmes evalues (LogisticRegression, RandomForest, XGBoost, LightGBM, SVM)
- **Optimisation** : GridSearchCV sur les hyperparametres cles

### Metriques cibles
- AUC-ROC (metrique principale)
- Precision / Recall / F1 sur la classe minoritaire (faillite)
- Matrice de confusion

## Structure du projet

```
.
├── staelle.ipynb              # Notebook principal — pipeline complet
├── models/
│   ├── lightgbm_best_model.pkl   # Modele entraine serialise
│   └── feature_names.pkl         # Noms des features selectionnees
├── training_data.csv          # Donnees d'entrainement
├── test_data.csv              # Donnees de test
├── templates/
│   └── index.html             # Interface web (Flask)
├── static/
│   └── style.css              # Style de l'interface
├── requirements.txt           # Dependances Python
└── Documentation_Faillite_Entreprise.pdf
```

## Deploiement

Application web Flask permettant de saisir les ratios financiers d'une entreprise et d'obtenir une prediction de risque en temps reel.

## Stack technique

- **Langage** : Python 3.12
- **ML** : LightGBM, Scikit-learn, XGBoost
- **Data** : Pandas, NumPy
- **Web** : Flask
- **Versioning** : Git / GitHub

## Auteur

**Jimmy Kenmo** — Data Science & Business Intelligence  
Universite de Dschang — Licence Professionnelle SIAD  
[Portfolio](https://portfolio-data-kenmo.netlify.app)
