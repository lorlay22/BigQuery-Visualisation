# Gérer temps
import time
# Utiliser expressions régulières pour reconnaissance de motifs
import re
# Journaliser événements et erreurs
import logging
import json
# Créer combinaisons d'éléments
import itertools
# Définir types données pour améliorer clarté
from typing import List, Dict, Any, Tuple
import colorsys # Pour générer des couleurs uniques
import numpy as np # Pour les calculs de similarité vectorielle

from google.cloud import bigquery
import networkx as nx # créer et manipuler structure graphe
from pyvis.network import Network # transformer graphe NetworkX en visualisation HTML interactive

# Import pour fonctionnalités IA
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import google.auth
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

# Configuration base pour journal d'événements
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# CLASSE CONFIGURATION 
class Config:
    # Remplacez par votre ID de projet Google Cloud
    PROJECT_ID = "votre-projet-gcp-ici"
    
    # Laissez vide pour traiter tous datasets, ou spécifiez une liste 
    DATASETS_TO_PROCESS_FILTER: List[str] = []

    # Modèles Vertex AI (non utilisés dans cette version mais conservés pour référence)
    VERTEX_AI_ENDPOINT = "us-central1-aiplatform.googleapis.com"
    TEXT_EMBEDDING_MODEL_NAME = "text-embedding-gecko"
    TEXT_GENERATION_MODEL_NAME = "text-bison"
    
    # Expressions régulières pour nettoyer noms colonnes et extraire concept principal
    CORE_NAME_PREFIXES_RE = re.compile(r'^(LINK_ID_|FK_ID_|ID_FK_|FK_|ID_|NUM_|REF_)', re.IGNORECASE)
    CORE_NAME_SUFFIXES_RE = re.compile(r'(_ID_LINK|_ID_FK_|_FK_ID|_ID|_FK|_KEY|_CODE|_NUM|_NUMBER|_REF)$', re.IGNORECASE)
    MIN_CORE_NAME_LENGTH = 3 # Longueur minimale du nom de base pour être considéré comme un concept
    
    # Liste de concepts à ignorer (mots courants qui ne représentent pas liens métier)
    CONCEPT_BLACKLIST: List[str] = ['DATE', 'USER', 'NAME', 'LABEL', 'TYPE', 'STATUS', 'DESCRIPTION', 'COMMENT', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND', 'TIMESTAMP']
    
    # Liste de concepts à toujours considérer comme clés, même s'ils ne respectent pas règles
    CONCEPT_WHITELIST: List[str] = ['VOTRE_CONCEPT_SPECIFIQUE'] # Ex: 'PRODUIT', 'CLIENT'
    
    # Mots-clés qualifiant comme une clé potentielle
    KEY_COLUMN_KEYWORDS: List[str] = ['_ID', '_KEY', '_CODE', '_REF', '_NUM']
    
    # STYLE ET APPARENCE DU GRAPHE
    # Définitions couleurs et styles pour différents éléments graphe
    COLUMN_COLOR = '#E0E0E0'     # Gris très clair pour colonnes simples
    KEY_COLUMN_COLOR = '#FFCCBC'    # Corail très clair pour clés
    LINK_COLOR = '#90A4AE'      # Gris-bleu moyen pour liens
    FONT_COLOR_LIGHT = '#424242'    # Texte foncé sur fonds clairs
    FONT_COLOR_DARK = '#263238'     # Texte très foncé

    # Distances et tailles pour physique et lisibilité
    COLUMN_LINK_LENGTH = 180      # Plus long pour étirer vue
    TABLE_LINK_LENGTH = 150
    DATASET_LINK_LENGTH = 200

    # Options pour librairie visualisation, contrôlent comportement physique du graphe
    PYVIS_OPTIONS = """
    {
      "nodes": {
        "font": { "size": 20, "face": "Roboto, Helvetica Neue, Helvetica, Arial, sans-serif", "color": "#263238" },
        "borderWidth": 1.5,
        "borderWidthSelected": 3,
        "shadow": { "enabled": true, "color": "rgba(0,0,0,0.2)", "size": 8, "x": 2, "y": 2 }
      },
      "edges": {
        "smooth": { "enabled": false },
        "width": 2,
        "color": { "inherit": "from", "highlight": "#03A9F4" },
        "font": { "size": 10, "color": "#424242", "background": "none", "strokeWidth": 0 }
      },
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.01,
          "springLength": 250,
          "springConstant": 0.05,
          "damping": 0.8,
          "avoidOverlap": 1.0
        },
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "fit": true
        }
      },
      "interaction": { "hover": true, "navigationButtons": true, "keyboard": true, "zoomView": true },
      "layout": { "improvedLayout": true }
    }
    """

class AISchemaEnhancer:
    def __init__(self, project_id: str, location: str):
        self.project_id = project_id
        self.location = location
        # aiplatform.init(project=self.project_id, location=self.location)

        # Utilise modèle local de sentence-transformers pour la recherche sémantique
        # Cela évite de dépendre d'une API externe pour cette fonctionnalité.
        try:
            self.model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
            logging.info("Modèle d'embedding local chargé. Prêt pour la recherche sémantique.")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle d'embedding: {e}. La recherche sémantique ne sera pas disponible.")
            self.model = None

        self.column_name_to_embedding: Dict[str, np.ndarray] = {}
        self.all_column_names: List[str] = []

    def generate_embeddings(self, schemas: Dict[str, List[Dict[str, Any]]]):
        """Génère un embedding pour chaque nom de colonne unique et les stocke."""
        if not self.model:
            return
        
        logging.info("Génération des embeddings pour tous les noms de colonnes...")
        unique_columns = sorted(list(set(col['name'] for table_id, columns in schemas.items() for col in columns)))
        
        self.all_column_names = unique_columns
        
        # modèle encode liste de noms en liste d'embeddings
        embeddings = self.model.encode(self.all_column_names)
        
        for i, name in enumerate(self.all_column_names):
            self.column_name_to_embedding[name] = embeddings[i]
        
        logging.info("Génération des embeddings terminée.")

    def find_most_similar_column_names(self, query: str, top_k: int = 5) -> List[str]:
        """Recherche les noms de colonnes les plus sémantiquement similaires à une requête donnée."""
        if not self.model or not self.column_name_to_embedding:
            return []
        
        query_embedding = self.model.encode([query])
        
        # Calcule similarité cosinus entre requête et noms de colonnes
        similarities = cosine_similarity(query_embedding, list(self.column_name_to_embedding.values()))
        
        # Trie et renvoie les top_k noms les plus similaires
        top_k_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        results = [self.all_column_names[i] for i in top_k_indices if similarities[0][i] > 0.5] # Seuil de similarité
        
        logging.info(f"Recherche sémantique pour '{query}': trouvés {results}")
        return results

# CLASSES TECHNIQUES (connexion à BQ et récupération schémas données)
class SchemaFetcher:
    # Initialise client BQ pour projet spécifié
    def __init__(self, project_id: str): self.project_id = project_id; self.client = self._init_client()
    def _init_client(self) -> bigquery.Client:
        # Initialisation du client BQ et teste connexion.
        # Si échec, enregistre erreur critique et stoppe programme.
        try: client = bigquery.Client(project=self.project_id); client.list_datasets(project=self.project_id, max_results=1); logging.info(f"Client BigQuery initialisé pour '{client.project}'."); return client
        except Exception as e: logging.critical(f"Échec init client BigQuery: {e}", exc_info=True); raise
    # Récupère schémas des tables des datasets
    # Utilise INFORMATION_SCHEMA de BQ pour obtenir noms de tables et de colonnes + leur type
    def get_all_schemas(self, dataset_filter: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        all_schemas: Dict[str, List[Dict[str, Any]]] = {}; datasets_to_process = []
        # Liste datasets et applique filtre si nécessaire.
        try:
            all_datasets = [ds.dataset_id for ds in self.client.list_datasets(self.project_id)]
            if not dataset_filter: datasets_to_process = all_datasets
            else: datasets_to_process = [ds for ds in all_datasets if ds in dataset_filter]
        except Exception as e: logging.error(f"Impossible de lister les datasets: {e}"); return {}
        logging.info(f"Récupération des schémas pour {len(datasets_to_process)} dataset(s).")
        # Pour chaque dataset, exécute requête SQL pour obtenir ses colonnes
        # Stocke infos dans dictionnaire `all_schemas`.
        for dataset_id in datasets_to_process:
            query = f"SELECT table_name, column_name, data_type FROM `{self.project_id}.{dataset_id}`.INFORMATION_SCHEMA.COLUMNS ORDER BY table_name, ordinal_position;"
            try:
                for row in self.client.query(query).result():
                    full_table_id = f"{dataset_id}.{row.table_name}"
                    if full_table_id not in all_schemas: all_schemas[full_table_id] = []
                    all_schemas[full_table_id].append({'name': row.column_name, 'type': row.data_type})
            except Exception as e: logging.error(f"Erreur sur dataset '{dataset_id}': {e}")
        logging.info(f"Récupération terminée. {len(all_schemas)} schémas trouvés.")
        return all_schemas

# Construction du graphe logique à partir des schémas de données (identifie relations et crée structure du réseau)
class GraphBuilder:
    # Initialise constructeur de graphes avec config et graphe NetworkX vide.
    def __init__(self, config: Config, ai_enhancer: AISchemaEnhancer):
        self.config = config
        self.graph = nx.DiGraph()
        self.ai_enhancer = ai_enhancer # L'objet AI
        # Initialise un générateur de couleurs pour les datasets/tables
        self.color_generator = self._create_color_generator()
        self.key_column_to_tables_map: Dict[str, List[str]] = {}

    @staticmethod
    def _create_color_generator():
        """Génère une séquence de couleurs distinctes en HSV pour les datasets et tables, avec des teintes neutres et variées."""
        num_colors_base = 40
        hue_offset = 0.5 / num_colors_base
        
        for i in range(num_colors_base):
            hue = i / num_colors_base
            saturation = 0.40
            value = 0.90
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            yield '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        for i in range(num_colors_base):
            hue = (i / num_colors_base) + hue_offset
            if hue > 1.0: hue -= 1.0
            saturation = 0.30
            value = 0.85
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            yield '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        yield from GraphBuilder._create_color_generator()

    @staticmethod
    def get_core_name(col_name: str, config: Config) -> str: 
        name_upper = col_name.upper()
        name_no_prefix = config.CORE_NAME_PREFIXES_RE.sub('', name_upper)
        name_no_suffix = config.CORE_NAME_SUFFIXES_RE.sub('', name_no_prefix)
        final_name = name_no_suffix.strip('_')
        return final_name if len(final_name) >= config.MIN_CORE_NAME_LENGTH else ""

    # Détermine si colonne est considérée comme clé selon règles définies dans Config.
    def _is_key_column(self, col_name: str, core_name: str) -> bool:
        if not core_name or core_name in self.config.CONCEPT_BLACKLIST: return False
        if core_name in self.config.CONCEPT_WHITELIST: return True
        col_name_upper = col_name.upper()
        return any(keyword in col_name_upper for keyword in self.config.KEY_COLUMN_KEYWORDS)

    # Construit graphe principal au niveau des datasets, ajoutant des liens basés sur les clés communes
    def build_for_drilldown(self, schemas: Dict[str, List[Dict[str, Any]]]) -> None:
        logging.info("Construction du graphe de HAUT NIVEAU pour l'exploration 'Drill-Down'...")
        key_column_index: Dict[str, List[str]] = {}
        tables_by_dataset: Dict[str, list] = {}
        
        # Pour stocker couleurs générées dynamiquement pour chaque dataset et table
        dataset_colors = {}
        table_colors = {}

        # Première passe pour attribuer des couleurs aux datasets et tables
        for full_table_id, columns in schemas.items():
            dataset_id = full_table_id.split('.')[0]
            if dataset_id not in tables_by_dataset:
                tables_by_dataset[dataset_id] = []
                dataset_colors[dataset_id] = next(self.color_generator)
            tables_by_dataset[dataset_id].append(full_table_id)
            
            table_colors[full_table_id] = next(self.color_generator)

            for column in columns:
                core_name = self.get_core_name(column['name'], self.config)
                column['is_key'] = self._is_key_column(column['name'], core_name)
                # Populate self.key_column_to_tables_map
                if core_name: # Only add if a valid core_name is found
                    if core_name not in self.key_column_to_tables_map:
                        self.key_column_to_tables_map[core_name] = []
                    # Avoid duplicates
                    if full_table_id not in self.key_column_to_tables_map[core_name]:
                        self.key_column_to_tables_map[core_name].append(full_table_id)

                if column['is_key']:
                    if core_name not in key_column_index: key_column_index[core_name] = []
                    key_column_index[core_name].append(full_table_id)
        
        # Ajoute datasets comme nœuds principaux au graphe, avec une couleur dynamique.
        for dataset_id in tables_by_dataset.keys():
            self.graph.add_node(dataset_id, label=dataset_id,
                                color={'background': dataset_colors[dataset_id],
                                       'border': '#424242',
                                       'highlight': {'background': dataset_colors[dataset_id], 'border': '#03A9F4'}},
                                shape='box',
                                size=40,
                                font={'color': self.config.FONT_COLOR_DARK, 'size':28},
                                type='dataset',
                                margin=15)

        # Crée liens entre datasets s'ils partagent colonnes clés.
        for core_name, table_ids in key_column_index.items():
            linked_datasets = {tid.split('.')[0] for tid in table_ids}
            if len(linked_datasets) > 1:
                for (source_ds, target_ds) in itertools.combinations(linked_datasets, 2):
                    if self.graph.has_edge(source_ds, target_ds):
                        self.graph[source_ds][target_ds]['weight'] += 1
                        self.graph[source_ds][target_ds]['title'] += f", {core_name}"
                        self.graph[source_ds][target_ds]['width'] = self.graph[source_ds][target_ds]['weight'] * 1.5
                    else:
                        self.graph.add_edge(source_ds, target_ds, weight=1, title=f"Lien via {core_name}",
                                            color=self.config.LINK_COLOR, width=1.5,
                                            dashes=True)
        
        # Stocke relations table-dataset et couleurs dynamiques dans schémas pour que JS puisse les utiliser.
        schemas['__metadata__'] = {
            'tables_by_dataset': tables_by_dataset,
            'dataset_colors': dataset_colors,
            'table_colors': table_colors,
            'key_column_to_tables_map': self.key_column_to_tables_map,
            #Stocke les noms de colonnes et leurs embeddings pour la recherche sémantique
            'ai_embeddings': self.ai_enhancer.column_name_to_embedding,
            'ai_column_names': self.ai_enhancer.all_column_names
        }
        logging.info(f"Graphe 'Drill-Down' construit: {self.graph.number_of_nodes()} datasets.")

# Génération visualisation HTML interactive
class Visualizer:
    # Initialise visualiseur avec config.
    def __init__(self, config: Config): self.config = config
    # Génère fichier HTML interactif contenant graphe et fonctionnalités JS.
    def generate_interactive_html(self, graph: nx.DiGraph, schemas: dict, filename_prefix: str):
        if not graph.nodes: logging.warning("Graphe vide. Visualisation annulée."); return
        logging.info("Génération de la visualisation interactive...")
        
        # Crée copie des métadonnées pour éviter de modifier l'original.
        metadata_for_js = schemas.get('__metadata__', {}).copy()
        
        # Convertit tableaux numpy des embeddings en listes Python pour sérialisation JSON.
        if 'ai_embeddings' in metadata_for_js:
            metadata_for_js['ai_embeddings'] = {
                k: v.tolist() for k, v in metadata_for_js['ai_embeddings'].items()
            }
        
        schemas_for_js = {k: v for k, v in schemas.items() if k != '__metadata__'}
        schema_json = json.dumps(schemas_for_js); metadata_json = json.dumps(metadata_for_js)
        
        # Prépare options de style pour le JS.
        style_config_json = json.dumps({
            'COLUMN_COLOR': self.config.COLUMN_COLOR,
            'KEY_COLUMN_COLOR': self.config.KEY_COLUMN_COLOR,
            'LINK_COLOR': self.config.LINK_COLOR,
            'FONT_COLOR_LIGHT': self.config.FONT_COLOR_LIGHT,
            'FONT_COLOR_DARK': self.config.FONT_COLOR_DARK,
            'COLUMN_LINK_LENGTH': self.config.COLUMN_LINK_LENGTH,
            'TABLE_LINK_LENGTH': self.config.TABLE_LINK_LENGTH
        })
        
        # code JS qui sera injecté dans le fichier HTML pour gérer toute l'interactivité
        # (Ce long bloc de code est intentionnel pour garder le HTML final autonome)
        js_code = f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
            body {{
                font-family: 'Roboto', sans-serif;
                margin: 0;
                overflow: hidden;
                background-color: #F0F2F5;
            }}
            .top-right-container {{
                position: absolute;
                top: 15px;
                right: 15px;
                z-index: 1000;
            }}
            .button-primary {{
                padding: 10px 18px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s ease, transform 0.1s ease;
                font-weight: 500;
                color: #FFFFFF;
                background-color: #78909C;
            }}
            .button-primary:hover {{
                background-color: #546E7A;
                transform: translateY(-1px);
            }}
            .button-secondary {{
                padding: 10px 18px;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                cursor: pointer;
                transition: background-color 0.3s ease, transform 0.1s ease;
                font-weight: 500;
                background-color: #B0BEC5;
                color: #37474F;
            }}
            .button-secondary:hover {{
                background-color: #90A4AE;
                transform: translateY(-1px);
            }}
            #search-container {{
                background: rgba(255,255,255,0.98);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                display: flex;
                gap: 10px;
                align-items: center;
                margin-bottom: 10px; /* Add space below search bar */
            }}
            #search-input {{
                width: 250px;
                padding: 10px 15px;
                border: 1px solid #E0E0E0;
                border-radius: 5px;
                font-size: 16px;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.03);
            }}
            #network_id {{
                border-radius: 8px;
                overflow: hidden;
                border: 1px solid #E0E0E0;
            }}

            #legend-container {{
                position: absolute;
                bottom: 15px;
                left: 15px;
                z-index: 1000;
                background: rgba(255,255,255,0.98);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                font-size: 14px;
                color: #343A40;
            }}
            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }}
            /* Specific styles for legend shapes */
            .legend-shape-box {{
                width: 25px;
                height: 25px;
                margin-right: 10px;
                border-radius: 4px;
                border: 1px solid rgba(0,0,0,0.1);
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .legend-shape-rounded-rect {{ /* New class for rounded rectangle in legend */
                width: 30px;
                height: 20px;
                margin-right: 10px;
                border-radius: 8px; /* High border-radius for rounded rectangle */
                border: 1px solid rgba(0,0,0,0.1);
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .legend-shape-circle {{
                width: 25px;
                height: 25px;
                margin-right: 10px;
                border-radius: 50%;
                border: 1px solid rgba(0,0,0,0.1);
                box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            }}
            .legend-label {{
                font-weight: 500;
            }}
            .legend-line {{
                width: 35px;
                height: 2px;
                margin-right: 10px;
                background-color: black;
                position: relative;
                top: -2px;
            }}
            .legend-line.dashed {{
                border-top: 2px dashed;
                background-color: transparent;
                width: 35px;
                height: 0;
            }}
            .legend-line-label {{
                font-style: italic;
                color: #6C757D;
            }}
            #attribute-info-panel {{ /* NEW: Style for the info panel */
                position: absolute;
                top: 15px;
                left: 15px;
                z-index: 1000;
                background: rgba(255,255,255,0.98);
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                max-height: 90%;
                overflow-y: auto;
                font-size: 14px;
                color: #343A40;
                width: 300px; /* Adjust width as needed */
                display: none; /* Hidden by default */
            }}
            #attribute-info-panel h3 {{
                margin-top: 0;
                color: #263238;
                border-bottom: 1px solid #ECEFF1;
                padding-bottom: 8px;
                margin-bottom: 15px;
            }}
            #attribute-info-panel ul {{
                list-style-type: none;
                padding: 0;
                margin: 0;
            }}
            #attribute-info-panel li {{
                padding: 6px 0;
                border-bottom: 1px dashed #ECEFF1;
            }}
            #attribute-info-panel li:last-child {{
                border-bottom: none;
            }}
            #close-attribute-info {{
                position: absolute;
                top: 8px;
                right: 8px;
                background: none;
                border: none;
                font-size: 20px;
                cursor: pointer;
                color: #78909C;
            }}
            #close-attribute-info:hover {{
                color: #37474F;
            }}
        </style>

        <div class="top-right-container">
            <button id="back-to-global-view-button" class="button-primary" style="display:none;">Vue Globale</button>
            <div id="search-container">
                <input type="text" id="search-input" placeholder="Rechercher dataset, table ou colonne...">
                <button id="search-button" class="button-primary">Rechercher</button>
                <button id="clear-search-button" class="button-secondary">Effacer</button>
            </div>
        </div>


        <div id="legend-container">
            <h3>Légende</h3>
            <div id="legend-items"></div>
        </div>

        <div id="attribute-info-panel"> <button id="close-attribute-info">&times;</button>
            <h3 id="attribute-info-title"></h3>
            <div id="attribute-info-content"></div>
        </div>

        <script type="text/javascript">
            const link = document.createElement('link');
            link.href = 'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap';
            link.rel = 'stylesheet';
            document.head.appendChild(link);

            const schemas = {schema_json};
            const metadata = {metadata_json};
            const styleConfig = {style_config_json};
            
            // This is the original data to restore the graph
            let originalNodesData = new vis.DataSet(network.body.data.nodes.get().filter(n => n.type === 'dataset'));
            let originalEdgesData = new vis.DataSet(network.body.data.edges.get());
            
            // Initial stabilisation - turn physics off after initial layout
            network.on("stabilizationIterationsDone", function () {{
                network.setOptions({{ physics: false }});
                console.log("Initial stabilization done, physics off.");
            }});

            // Function to temporarily activate physics for layout adjustment
            function performLayoutStabilization(nodesToFit, iterations = 750, fitAnimationDuration = 500) {{
                network.setOptions({{ physics: true }});
                console.log("Physics enabled for layout adjustment.");
                network.redraw(); // Force a redraw to apply new physics

                setTimeout(() => {{
                    network.setOptions({{ physics: false }});
                    console.log("Physics disabled after layout adjustment.");
                    if (nodesToFit && nodesToFit.length > 0) {{
                        network.fit({{ nodes: nodesToFit, animation: {{"duration": fitAnimationDuration, "easing": "easeOutQuart"}} }});
                    }} else {{
                        network.fit({{ animation: {{"duration": fitAnimationDuration, "easing": "easeOutQuart"}} }});
                    }}
                }}, iterations); 
            }}

            // NEW: get_core_name function in JS (mirrors Python logic)
            function getCoreNameJS(col_name) {{
                const CORE_NAME_PREFIXES_RE = /^(LINK_ID_|FK_ID_|ID_FK_|FK_|ID_|NUM_|REF_)/i;
                const CORE_NAME_SUFFIXES_RE = /(_ID_LINK|_ID_FK_|_FK_ID|_ID|_FK|_KEY|_CODE|_NUM|_NUMBER|_REF)$/i;
                const MIN_CORE_NAME_LENGTH = 3;

                let name_upper = col_name.toUpperCase();
                let name_no_prefix = name_upper.replace(CORE_NAME_PREFIXES_RE, '');
                let name_no_suffix = name_no_prefix.replace(CORE_NAME_SUFFIXES_RE, '');
                let final_name = name_no_suffix.replace(/^_|_$/g, ''); // Strip leading/trailing underscores
                return final_name.length >= MIN_CORE_NAME_LENGTH ? final_name : "";
            }}

            function showFocusView(datasetId) {{
                // Store the current graph state before changing it
                originalNodesData = new vis.DataSet(network.body.data.nodes.get());
                originalEdgesData = new vis.DataSet(network.body.data.edges.get());

                const nodesToRemove = originalNodesData.get().filter(n => n.id !== datasetId && n.type === 'dataset');
                network.body.data.nodes.remove(nodesToRemove);
                
                // Expand the selected dataset to show its tables
                expandDataset(datasetId);

                // Show the "Back" button
                document.getElementById('back-to-global-view-button').style.display = 'inline-block';
            }}
            
            function showGlobalView() {{
                // Restore initial nodes and edges
                network.setData({{ nodes: originalNodesData, edges: originalEdgesData }});
                
                // Re-enable physics for stabilization
                network.setOptions({{ physics: true }});
                
                // Hide the "Back" button
                document.getElementById('back-to-global-view-button').style.display = 'none';
                
                performLayoutStabilization();
            }}
            
            document.getElementById('back-to-global-view-button').addEventListener('click', showGlobalView);


            function expandDataset(datasetId) {{
                const tables = metadata.tables_by_dataset[datasetId];
                if (!tables) return;
                
                // First, contract any other expanded datasets
                const expandedDatasets = network.body.data.nodes.get().filter(n => n.type === 'dataset' && n.expanded);
                expandedDatasets.forEach(d => contractDataset(d.id));

                let nodesToAdd = [];
                let edgesToAdd = [];
                for (const tableId of tables) {{
                    const tableColor = metadata.table_colors[tableId];
                    nodesToAdd.push({{
                        id: tableId,
                        label: tableId.split('.')[1],
                        title: tableId,
                        color: {{"background": tableColor,
                                         "border": "#546E7A", /* Bord neutre plus foncé pour les tables */
                                         "highlight": {{"background": tableColor, "border": "#03A9F4"}}}},
                        shape: 'box', /* Table shape is 'box' */
                        shapeProperties: {{ "borderRadius": 10 }}, /* Rounded corners for tables */
                        font: {{"color": styleConfig.FONT_COLOR_DARK, "size": 22}},
                        type: 'table',
                        margin: 15, /* Increased margin for tables */
                        physics: true
                    }});
                    edgesToAdd.push({{
                        from: datasetId,
                        to: tableId,
                        color: {{"color": "#B0BEC5", "highlight": "#03A9F4"}}, /* Lien table-dataset plus doux */
                        arrows: 'to',
                        width: 1.5,
                        length: styleConfig.TABLE_LINK_LENGTH,
                        physics: true
                    }});
                }}
                network.body.data.nodes.add(nodesToAdd);
                network.body.data.edges.add(edgesToAdd);
                network.body.data.nodes.update({{ id: datasetId, expanded: true }});
                
                // Adjust physics for expanded dataset (more repulsion, more iterations)
                network.setOptions({{
                    physics: {{
                        forceAtlas2Based: {{
                            gravitationalConstant: -1500, 
                            springLength: 200,             
                            springConstant: 0.05,        
                            avoidOverlap: 1.0            
                        }},
                        stabilization: {{ iterations: 500 }}
                    }}
                }});
                performLayoutStabilization([datasetId].concat(tables), 750); // More iterations for expanded view
            }}

            function contractDataset(datasetId) {{
                const tables = metadata.tables_by_dataset[datasetId];
                if (!tables) return;
                let nodesToRemove = [];
                for (const tableId of tables) {{
                    const tableNode = network.body.data.nodes.get(tableId);
                    if(tableNode && tableNode.columns_expanded) {{ contractTable(tableId); }}
                    nodesToRemove.push(tableId);
                }}
                network.body.data.nodes.remove(nodesToRemove);
                network.body.data.nodes.update({{ id: datasetId, expanded: false }});
                
                // Reset physics to default (less aggressive) for contracted dataset view
                network.setOptions({{
                    physics: {{
                        forceAtlas2Based: {{
                            gravitationalConstant: -2000,  
                            centralGravity: 0.01,      
                            springLength: 250,             
                            springConstant: 0.05,      
                            damping: 0.8,
                            avoidOverlap: 1.0            
                        }},
                        stabilization: {{ iterations: 500 }}
                    }}
                }});
                performLayoutStabilization([datasetId]);
            }}

            function expandTable(tableId) {{
                const columns = schemas[tableId]; if (!columns) return;
                let nodesToAdd = [];
                let edgesToAdd = [];
                for (const col of columns) {{
                    const colId = tableId + '.' + col.name;
                    nodesToAdd.push({{
                        id: colId,
                        label: col.name,
                        title: 'Type: ' + col.type,
                        color: {{"background": col.is_key ? styleConfig.KEY_COLUMN_COLOR : styleConfig.COLUMN_COLOR,
                                         "border": col.is_key ? '#D32F2F' : '#78909C'}}, /* Bords pour colonnes (plus neutres/gris) */
                        shape: 'ellipse', /* Column shape is 'ellipse' */
                        font: {{"color": styleConfig.FONT_COLOR_DARK, "size": 18}},
                        type: 'column',
                        size: 20,
                        margin: 5,
                        physics: true
                    }});
                    edgesToAdd.push({{
                        from: tableId,
                        to: colId,
                        length: styleConfig.COLUMN_LINK_LENGTH,
                        color: {{"color": "#CFD8DC", "highlight": "#03A9F4"}}, /* Lien table-colonne */
                        width: 1,
                        physics: true
                    }});
                }}
                network.body.data.nodes.add(nodesToAdd);
                network.body.data.edges.add(edgesToAdd);
                network.body.data.nodes.update({{ id: tableId, columns_expanded: true }});
                
                // Adjust physics for expanded table (even more repulsion, even more iterations)
                network.setOptions({{
                    physics: {{
                        forceAtlas2Based: {{
                            gravitationalConstant: -1800, 
                            springLength: 150,             
                            springConstant: 0.03,        
                            avoidOverlap: 1.0            
                        }},
                        stabilization: {{ iterations: 750 }}
                    }}
                }});
                performLayoutStabilization([tableId].concat(columns.map(c => tableId + '.' + c.name)), 1000); // More iterations for columns
            }}

            function contractTable(tableId) {{
                const columns = schemas[tableId]; if (!columns) return;
                let nodesToRemove = columns.map(col => tableId + '.' + col.name);
                network.body.data.nodes.remove(nodesToRemove);
                network.body.data.nodes.update({{ id: tableId, columns_expanded: false }});
                
                // Reset physics for contracted table (less aggressive)
                network.setOptions({{
                    physics: {{
                        forceAtlas2Based: {{
                            gravitationalConstant: -1500, 
                            springLength: 200,             
                            springConstant: 0.05,        
                            avoidOverlap: 1.0            
                        }},
                        stabilization: {{ iterations: 500 }}
                    }}
                }});
                performLayoutStabilization([tableId]);
            }}

            // NEW: Function to show attribute info panel
            function showAttributeInfo(attributeName, tablesList, coreName) {{
                const panel = document.getElementById('attribute-info-panel');
                const title = document.getElementById('attribute-info-title');
                const content = document.getElementById('attribute-info-content');

                title.textContent = `Tables contenant l'attribut "${{attributeName}}"`;
                if (coreName) {{
                    title.textContent += ` (Base: ${{coreName}})`;
                }}
                content.innerHTML = ''; // Clear previous content

                if (tablesList && tablesList.length > 0) {{
                    const ul = document.createElement('ul');
                    tablesList.forEach(table => {{
                        const li = document.createElement('li');
                        li.textContent = table;
                        ul.appendChild(li);
                    }});
                    content.appendChild(ul);
                }} else {{
                    content.textContent = "Aucune autre table trouvée avec cet attribut.";
                }}
                panel.style.display = 'block';
            }}

            // NEW: Function to hide attribute info panel
            function hideAttributeInfo() {{
                document.getElementById('attribute-info-panel').style.display = 'none';
            }}

            document.getElementById('close-attribute-info').addEventListener('click', hideAttributeInfo);


            network.on("click", function (params) {{
                if (params.nodes.length > 0) {{
                    const nodeId = params.nodes[0];
                    const node = network.body.data.nodes.get(nodeId);
                    if (!node) return;

                    if (node.type === 'dataset') {{
                        const isExpanded = node.expanded || false;
                        if (isExpanded) {{
                            contractDataset(nodeId);
                        }} else {{
                            showFocusView(nodeId);
                        }}
                    }}
                    else if (node.type === 'table') {{
                        const isExpanded = node.columns_expanded || false;
                        if(isExpanded) {{ contractTable(nodeId); }} else {{ expandTable(nodeId); }}
                    }}
                    else if (node.type === 'column') {{ // NEW: Handle clicks on columns
                        hideAttributeInfo(); // Hide any existing panel
                        const [datasetId, tableName, columnName] = nodeId.split('.');
                        const fullTableName = `${{datasetId}}.${{tableName}}`;
                        const columnInfo = schemas[fullTableName].find(col => col.name === columnName);
                        
                        if (columnInfo) {{
                            const coreName = getCoreNameJS(columnName);
                            if (coreName) {{
                                const tablesWithSameAttribute = metadata.key_column_to_tables_map[coreName];
                                if (tablesWithSameAttribute) {{
                                    // Filter out the current table from the list for display purposes
                                    const filteredTables = tablesWithSameAttribute.filter(table => table !== fullTableName);
                                    showAttributeInfo(columnName, filteredTables, coreName);
                                }} else {{
                                    alert(`L'attribut '${{columnName}}' (nom de base: '${{coreName}}') n'est pas trouvé dans d'autres tables.`);
                                }}
                            }} else {{
                                alert(`Impossible de déterminer le nom de base pour l'attribut '${{columnName}}'.`);
                            }}
                        }}
                    }}
                }}
            }});

            document.getElementById('search-button').addEventListener('click', function() {{
                const searchTerm = document.getElementById('search-input').value; // Keep original case for AI
                searchAndHighlightNode(searchTerm);
            }});

            document.getElementById('search-input').addEventListener('keypress', function(e) {{
                if (e.key === 'Enter') {{
                    const searchTerm = document.getElementById('search-input').value; // Keep original case for AI
                    searchAndHighlightNode(searchTerm);
                }}
            }});

            document.getElementById('clear-search-button').addEventListener('click', function() {{
                clearSearch();
            }});

            function getSemanticMatches(query, allColumnNames, columnEmbeddings, threshold = 0.5) {{
                // Placeholder for real AI logic
                // In a real scenario, this would call a server-side endpoint
                // We will use a simple function to simulate the behavior for now
                const results = [];
                for (const name of allColumnNames) {{
                    const coreName = getCoreNameJS(name);
                    const queryCoreName = getCoreNameJS(query);
                    if (coreName && queryCoreName && coreName === queryCoreName) {{
                        results.push(name);
                    }}
                }}
                return results;
            }}


            async function searchAndHighlightNode(searchTerm) {{
                clearSearch();
                hideAttributeInfo();
                document.getElementById('back-to-global-view-button').style.display = 'inline-block';

                if (!searchTerm) {{
                    return;
                }}
                
                // Utilise la recherche sémantique pour trouver les noms de colonnes pertinents
                const semanticMatches = getSemanticMatches(searchTerm, metadata.ai_column_names, metadata.ai_embeddings);
                const strictCoreNameMatch = getCoreNameJS(searchTerm).toUpperCase();

                let tablesWithMatches = new Set();
                let matchingColumnsInfo = new Map(); // Store column info for linking

                for (const tableId in schemas) {{
                    const columns = schemas[tableId];
                    for (const col of columns) {{
                        const colCoreName = getCoreNameJS(col.name).toUpperCase();
                        
                        // Check for strict core name match
                        const isCoreNameMatch = strictCoreNameMatch && colCoreName === strictCoreNameMatch;
                        
                        // Check for semantic match
                        const isSemanticMatch = semanticMatches.includes(col.name);

                        if (isCoreNameMatch || isSemanticMatch) {{
                            tablesWithMatches.add(tableId);
                            
                            const colId = tableId + '.' + col.name;
                            matchingColumnsInfo.set(colId, {{
                                ...col,
                                tableId: tableId,
                                coreName: colCoreName
                            }});
                        }}
                    }}
                }}
                
                if (tablesWithMatches.size > 0) {{
                    let nodesToDisplay = [];
                    let edgesToDisplay = [];
                    let foundColumnIds = [];
                    let foundKeyColumnIds = [];

                    // Add datasets and tables
                    const datasetsToDisplay = new Set();
                    tablesWithMatches.forEach(tableId => {{
                        const datasetId = tableId.split('.')[0];
                        datasetsToDisplay.add(datasetId);
                    }});

                    datasetsToDisplay.forEach(datasetId => {{
                        const datasetNode = originalNodesData.get(datasetId);
                        if (datasetNode) {{
                            nodesToDisplay.push(datasetNode);
                        }}
                    }});

                    tablesWithMatches.forEach(tableId => {{
                        const tableNode = {{
                            id: tableId,
                            label: tableId.split('.')[1],
                            title: tableId,
                            color: {{
                                "background": metadata.table_colors[tableId],
                                "border": "#546E7A",
                                "highlight": {{
                                    "background": metadata.table_colors[tableId],
                                    "border": "#03A9F4"
                                }}
                            }},
                            shape: 'box',
                            shapeProperties: {{
                                "borderRadius": 10
                            }},
                            font: {{
                                "color": styleConfig.FONT_COLOR_DARK,
                                "size": 22
                            }},
                            type: 'table',
                            margin: 15,
                            physics: true
                        }};
                        nodesToDisplay.push(tableNode);
                        edgesToDisplay.push({{
                            from: tableId.split('.')[0],
                            to: tableId,
                            color: {{
                                "color": "#B0BEC5",
                                "highlight": "#03A9F4"
                            }},
                            arrows: 'to',
                            width: 1.5,
                            length: styleConfig.TABLE_LINK_LENGTH,
                            physics: true
                        }});
                    }});

                    // Add matching columns and their links to tables
                    matchingColumnsInfo.forEach((col, colId) => {{
                        foundColumnIds.push(colId);
                        if (col.is_key) {{
                            foundKeyColumnIds.push(colId);
                        }}
                        
                        nodesToDisplay.push({{
                            id: colId,
                            label: col.name,
                            title: `Type: ${{col.type}} (Attribut de base: ${{col.coreName}})`,
                            color: {{
                                "background": col.is_key ? styleConfig.KEY_COLUMN_COLOR : styleConfig.COLUMN_COLOR,
                                "border": '#1565C0',
                                "highlight": {{ "background": col.is_key ? styleConfig.KEY_COLUMN_COLOR : styleConfig.COLUMN_COLOR, "border": "#03A9F4" }}
                            }},
                            borderWidth: 3,
                            shape: 'ellipse',
                            font: {{
                                "color": styleConfig.FONT_COLOR_DARK,
                                "size": 18
                            }},
                            type: 'column',
                            size: 20,
                            margin: 5,
                            physics: true
                        }});

                        edgesToDisplay.push({{
                            from: col.tableId,
                            to: colId,
                            length: styleConfig.COLUMN_LINK_LENGTH,
                            color: {{
                                "color": "#CFD8DC",
                                "highlight": "#03A9F4"
                            }},
                            width: 1,
                            physics: true
                        }});
                    }});
                    
                    // NEW: Add direct links between *found key* columns only
                    if (foundKeyColumnIds.length > 1) {{
                        for (let i = 0; i < foundKeyColumnIds.length; i++) {{
                            for (let j = i + 1; j < foundKeyColumnIds.length; j++) {{
                                const nodeA = foundKeyColumnIds[i];
                                const nodeB = foundKeyColumnIds[j];
                                edgesToDisplay.push({{
                                    from: nodeA,
                                    to: nodeB,
                                    title: `Lien via attribut clé partagé '${{getCoreNameJS(matchingColumnsInfo.get(nodeA).name)}}'`,
                                    color: {{
                                        "color": '#D32F2F',
                                        "highlight": "#03A9F4"
                                    }},
                                    width: 3,
                                    dashes: true,
                                    physics: true
                                }});
                            }}
                        }}
                    }}

                    // Mise à jour du graphe
                    network.setData({{
                        nodes: nodesToDisplay,
                        edges: edgesToDisplay
                    }});

                    // Ajustement de la physique et du zoom
                    network.setOptions({{ physics: true }});
                    const nodesToFit = [...datasetsToDisplay, ...tablesWithMatches, ...foundColumnIds];
                    network.fit({{ nodes: nodesToFit, animation: {{ duration: 1500, easing: 'easeInOutQuad' }} }});
                    showAttributeInfo(searchTerm, Array.from(tablesWithMatches), strictCoreNameMatch);
                }} else {{
                    alert(`Aucun attribut (ou attribut sémantiquement similaire) trouvé pour '${{searchTerm}}'.`);
                }}
            }}


            function clearSearch() {{
                network.unselectAll();
                hideAttributeInfo();
                showGlobalView();
            }}

            // --- Légende ---
            function populateLegend() {{
                const legendItemsDiv = document.getElementById('legend-items');
                legendItemsDiv.innerHTML = '';

                // Dataset: Standard Box
                const firstDatasetId = Object.keys(metadata.dataset_colors)[0];
                const datasetBorderColor = '#424242';
                const datasetColor = firstDatasetId ? metadata.dataset_colors[firstDatasetId] : '#E0E0E0';
                legendItemsDiv.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-shape-box" style="background-color: ${{datasetColor}}; border-color: ${{datasetBorderColor}};"></div>
                        <span class="legend-label">Dataset (Rectangle)</span>
                    </div>
                `;

                // Table: Rounded Rectangle
                const firstTableId = Object.keys(metadata.table_colors)[0];
                const tableBorderColor = '#546E7A';
                const tableColor = firstTableId ? metadata.table_colors[firstTableId] : '#CFD8DC';
                legendItemsDiv.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-shape-rounded-rect" style="background-color: ${{tableColor}}; border-color: ${{tableBorderColor}};"></div>
                        <span class="legend-label">Table (Rectangle Arrondi)</span>
                    </div>
                `;

                // Colonne Clé: Circle
                legendItemsDiv.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-shape-circle" style="background-color: ${{styleConfig.KEY_COLUMN_COLOR}}; border-color: #D32F2F;"></div>
                        <span class="legend-label">Colonne Clé (Cercle)</span>
                    </div>
                `;

                // Colonne Simple: Circle
                legendItemsDiv.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-shape-circle" style="background-color: ${{styleConfig.COLUMN_COLOR}}; border-color: #78909C;"></div>
                        <span class="legend-label">Colonne Simple (Cercle)</span>
                    </div>
                `;
                // NEW: Search result highlight
                legendItemsDiv.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-shape-circle" style="background-color: ${{styleConfig.COLUMN_COLOR}}; border: 3px solid #1565C0;"></div>
                        <span class="legend-label">Résultat de Recherche</span>
                    </div>
                `;

                // Lien inter-datasets
                legendItemsDiv.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-line dashed" style="border-color: ${{styleConfig.LINK_COLOR}};"></div>
                        <span class="legend-line-label">Lien Inter-Datasets</span>
                    </div>
                `;

                // Lien table-dataset / table-colonne
                legendItemsDiv.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-line" style="background-color: #B0BEC5;"></div>
                        <span class="legend-line-label">Lien Table/Colonne</span>
                    </div>
                `;

                // NEW: Link between matching columns
                legendItemsDiv.innerHTML += `
                    <div class="legend-item">
                        <div class="legend-line dashed" style="border-color: #D32F2F;"></div>
                        <span class="legend-line-label">Lien Attribut Clé Partagé</span>
                    </div>
                `;
            }}

            setTimeout(populateLegend, 700);

        </script>
        """
        
        # Crée l'objet réseau Pyvis, applique options de visualisation.
        net = Network(notebook=False, height="950px", width="100%", directed=True, bgcolor="#F0F2F5", cdn_resources='remote')
        net.from_nx(graph)
        net.set_options(self.config.PYVIS_OPTIONS)
        
        # Génère nom de fichier unique avec un horodatage.
        ts = time.strftime("%Y%m%d-%H%M%S"); filename = f"{filename_prefix}_{ts}_SCHEMA_MAP.html"
        # Sauvegarde graphe dans fichier HTML.
        net.save_graph(filename)
        # Ré-ouvre fichier HTML pour y injecter code JS personnalisé
        with open(filename, 'r+', encoding='utf-8') as f: 
            content = f.read()
            # Remplace fin du body par code JS avant balise </body>
            f.seek(0, 0)
            f.write(content.replace('</body>', js_code + '</body>'))
        logging.info(f"Visualisation générée : '{filename}'.")


# Exécution du script
def main():
    script_start_time = time.time(); logging.info("===== DÉBUT DU SCRIPT DE CARTOGRAPHIE DE SCHÉMA ====="); config = Config()
    try:
        # 1. Instancier l'AI Enhancer
        ai_enhancer = AISchemaEnhancer(project_id=config.PROJECT_ID, location="us-central1")
        
        # 2. Récupération schémas via SchemaFetcher.
        fetcher = SchemaFetcher(project_id=config.PROJECT_ID); schemas = fetcher.get_all_schemas(dataset_filter=config.DATASETS_TO_PROCESS_FILTER)
        if not schemas: logging.warning("Aucun schéma trouvé. Arrêt du script."); return
        
        # 3. Génération embeddings IA ( recherche sémantique)
        ai_enhancer.generate_embeddings(schemas)
        
        # 4. Construction graphe logique via GraphBuilder.
        builder = GraphBuilder(config=config, ai_enhancer=ai_enhancer); builder.build_for_drilldown(schemas)
        
        # 5. Génération visualisation HTML interactive via Visualizer.
        visualizer = Visualizer(config=config)
        fname_suffix = "full" if not config.DATASETS_TO_PROCESS_FILTER else "filtered"
        # Nom de fichier anonymisé pour ne pas exposer l'ID de projet
        filename_prefix = f"schema_graph_{fname_suffix}"
        visualizer.generate_interactive_html(builder.graph, schemas, filename_prefix)
    # Gère erreurs critiques
    except Exception as e: logging.critical(f"Erreur critique: {e}", exc_info=True)
    # S'exécute toujours
    # Affiche durée totale d'exécution et des instructions pour l'utilisateur.
    finally:
        duration = time.time() - script_start_time; logging.info(f"===== SCRIPT TERMINÉ en {duration:.2f} secondes. =====")
        print("✅ Fichier HTML généré avec succès !")
        print("   -> Le graphe affiche les datasets. Cliquez sur un dataset pour explorer ses tables, puis sur une table pour voir ses colonnes.")
        print("   -> **Nouveau !** Cliquez sur une colonne pour voir dans quelles autres tables cet attribut apparaît.")
        print("   -> **Recherche sémantique** : Utilisez la barre de recherche pour trouver toutes les tables et colonnes qui partagent un sens commun.")
        print("\n✨ Ouvrez le fichier .html généré dans votre navigateur pour commencer l'exploration !")


# Point de démarrage du script
if __name__ == '__main__':
    main()
