# Visualiseur de Schéma BigQuery avec IA

Ce projet est un outil en Python qui se connecte à un projet Google Cloud, analyse les schémas de tous les datasets BigQuery, et génère une cartographie des données sous forme de graphe interactif. Il utilise une combinaison d'analyse heuristique et d'intelligence artificielle (embeddings sémantiques) pour déduire et visualiser les relations entre les tables.

L'objectif est de fournir une vue d'ensemble claire et explorable de leur patrimoine de données, facilitant ainsi la découverte de données et la compréhension du lignage.


## ✨ Fonctionnalités Principales

-   **Connexion Automatique à BigQuery** : Récupère les schémas de tous les datasets d'un projet GCP.
-   **Analyse Heuristique** : Utilise des expressions régulières et des règles métier pour identifier les colonnes qui agissent comme des clés (clés primaires/étrangères).
-   **Construction de Graphe Logique** : Modélise les relations entre les datasets, les tables et les colonnes en utilisant la bibliothèque `NetworkX`.
-   **Visualisation Interactive HTML** : Génère un fichier HTML unique et autonome avec une visualisation dynamique et explorable grâce à `Pyvis`.
-   **Interface "Drill-Down"** :
    1.  Commencez par une vue globale des datasets.
    2.  Cliquez sur un dataset pour voir les tables qu'il contient.
    3.  Cliquez sur une table pour afficher toutes ses colonnes.
    4.  Cliquez sur une colonne clé pour voir où elle est utilisée dans d'autres tables.
-   **Recherche Sémantique (IA)** : Une barre de recherche permet de trouver des colonnes par leur sens grâce à des embeddings de texte générés par `Sentence-Transformers`. Par exemple, chercher "identifiant client" peut trouver des colonnes nommées `customer_id`, `id_cli`, ou `user_num`.

## 🛠️ Technologies Utilisées

-   **Python 3**
-   **Google Cloud SDK** (pour l'API BigQuery)
-   **NetworkX** : Pour la création et la manipulation de la structure du graphe.
-   **Pyvis** : Pour la conversion du graphe NetworkX en une visualisation HTML/JS interactive.
-   **Sentence-Transformers** : Pour la création de "vector embeddings" des noms de colonnes afin de permettre la recherche sémantique.
-   **Numpy & Scikit-learn** : Pour les calculs de similarité cosinus entre les vecteurs.

## 🚀 Comment l'utiliser

### Prérequis

-   Avoir un compte Google Cloud avec un projet BigQuery contenant des données.
-   Avoir Python 3.7+ installé.
-   Avoir authentifié `gcloud` sur votre machine locale.

### Installation

1.  **Clonez ce dépôt :**
    ```bash
    git clone [https://github.com/votre-pseudo/votre-repo.git](https://github.com/votre-pseudo/votre-repo.git)
    cd votre-repo
    ```

2.  **Installez les dépendances :**
    (Il est recommandé d'utiliser un environnement virtuel)
  pip install google-cloud-bigquery networkx pyvis sentence-transformers scikit-learn numpy

3.  **Authentifiez-vous auprès de Google Cloud (si ce n'est pas déjà fait) :**
    ```bash
    gcloud auth application-default login
    ```

### Configuration

Ouvrez le script principal et modifiez la classe `Config` :

-   **`PROJECT_ID`**: Remplacez `"votre-projet-gcp-ici"` par l'ID de votre projet Google Cloud.
-   **`DATASETS_TO_PROCESS_FILTER`** (Optionnel) : Si vous souhaitez analyser seulement quelques datasets, listez-les ici (ex: `["sales", "marketing"]`).

### Lancement

Exécutez simplement le script depuis votre terminal :

```bash
python votre_script.py
```

Le script va se connecter à BigQuery, analyser les schémas, et générer un fichier `schema_graph_... .html` dans le même dossier. Ouvrez ce fichier dans votre navigateur pour explorer la cartographie !
