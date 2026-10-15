import json
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent

# Tous les fichiers .json de paramétrage (modèles de fiches, vocabulaires, observateurs, taxons par
# région...) vivent dans le répertoire config/. Ce module s'appelle volontairement settings.py (et non
# config.py) pour ne pas être confondu avec ce répertoire.
CONFIG_DIR = _ROOT / 'config'


def _load_json(path, default):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


# ----------------------------- Détection des cases vides -----------------------------
# Sur une case réellement vide (papier uniforme, léger bruit/ombre), un seuillage Otsu classique
# force quand même une coupure ~bimodale et peut faire croire à de l'encre : on regarde d'abord
# le contraste (écart-type) et on ne considère la case comme "encrée" que s'il est suffisant, puis
# on exige une tache d'une taille minimale (relative à la case) pour éviter le bruit résiduel.
# Ces seuils sont indépendants du modèle de fiche (ils dépendent de la qualité de la photo, pas
# de la disposition du formulaire).
BLANK_STD_THRESHOLD = 10.0
BLANK_INK_RATIO = 0.008
BLANK_MIN_AREA_RATIO = 0.0012


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    width_mm: float


# ----------------------------- Modèles de fiches -----------------------------
# La disposition (nombre/largeur des colonnes, nombre de lignes, hauteurs) dépend du modèle de
# fiche papier utilisé : elle vient de config/modeles_fiches.json plutôt que d'être codée en dur ici,
# pour pouvoir gérer plusieurs modèles (CACHI_2024, CACCHI_ancien, GMB_29, ...) sans toucher au code.
TEMPLATES_PATH = CONFIG_DIR / 'modeles_fiches.json'


class FormTemplate:
    """Disposition géométrique d'un modèle de fiche papier (une entrée de modeles_fiches.json)."""

    def __init__(self, name, data):
        self.name = name
        self.canonical_width = int(data['canonical_width'])
        self.canonical_height = int(data['canonical_height'])
        self.form_width_mm = float(data['form_width_mm'])
        self.form_height_mm = float(data['form_height_mm'])
        self.header_height_mm = float(data['header_height_mm'])
        self.row_height_mm = float(data['row_height_mm'])
        self.n_rows = int(data['n_rows'])
        self.column_gap_mm = float(data.get('column_gap_mm', 0.1))
        self.row_gap_mm = float(data.get('row_gap_mm', 0.1))
        self.columns = [Column(c['key'], c['label'], float(c['width_mm'])) for c in data['columns']]
        # Positions X optionnelles : x_mm = bord gauche de la colonne.
        # Si absent, la colonne commence juste après la précédente.
        self.column_x_mm = []
        cursor = 0.0
        for c in data['columns']:
            x = float(c.get('x_mm', cursor))
            self.column_x_mm.append(x)
            cursor = x + float(c['width_mm'])
        self._validate()

    def _validate(self):
        # Avec x_mm explicite, les colonnes n'ont plus besoin d'être contiguës.
        # Sans x_mm, on retrouve le comportement historique par cumul des largeurs.
        cursor = 0.0
        for i, c in enumerate(self.columns):
            x = self.column_x_mm[i]
            if x < -1e-6 or x + c.width_mm > self.form_width_mm + 1e-6:
                raise ValueError(f"Modèle '{self.name}' : colonne '{c.key}' hors formulaire "
                                 f"(x={x}, largeur={c.width_mm}, largeur formulaire={self.form_width_mm})")
            if i and x < self.column_x_mm[i-1] + self.columns[i-1].width_mm - 1e-6:
                raise ValueError(f"Modèle '{self.name}' : chevauchement des colonnes '{self.columns[i-1].key}' "
                                 f"et '{c.key}'")
            cursor = max(cursor, x + c.width_mm)
        if abs(cursor - self.form_width_mm) >= 1e-6:
            raise ValueError(f"Modèle '{self.name}' : la dernière colonne s'arrête à {cursor} mm "
                             f"mais form_width_mm vaut {self.form_width_mm} mm")
        total_h = self.header_height_mm + self.n_rows * self.row_height_mm
        if abs(total_h - self.form_height_mm) >= 1e-6:
            raise ValueError(f"Modèle '{self.name}' : header + lignes ({total_h}) "
                              f"!= form_height_mm ({self.form_height_mm})")


def _load_templates():
    raw = _load_json(TEMPLATES_PATH, {})
    templates = {}
    for name, data in raw.items():
        try:
            templates[name] = FormTemplate(name, data)
        except Exception as e:
            print(f"[modeles_fiches.json] modèle '{name}' ignoré : {e}")
    return templates


TEMPLATES = _load_templates()
DEFAULT_TEMPLATE_NAME = next(iter(TEMPLATES), None)


class ActiveConfig:
    """Représente le modèle de fiche actuellement sélectionné dans l'application. C'est un objet
    MUTABLE unique (voir `active` plus bas) : changer de modèle (set_template) met à jour tous ses
    attributs en place, donc tout code qui a gardé une référence à `active` voit immédiatement la
    nouvelle disposition, sans avoir besoin de ré-importer quoi que ce soit."""

    def __init__(self, name):
        if name is None:
            raise SystemExit("Aucun modèle de fiche défini dans config/modeles_fiches.json.")
        self.set_template(name)

    def set_template(self, name):
        t = TEMPLATES[name]
        self.template_name = name
        self.CANONICAL_WIDTH = t.canonical_width
        self.CANONICAL_HEIGHT = t.canonical_height
        self.FORM_WIDTH_MM = t.form_width_mm
        self.FORM_HEIGHT_MM = t.form_height_mm
        self.HEADER_HEIGHT_MM = t.header_height_mm
        self.ROW_HEIGHT_MM = t.row_height_mm
        self.N_ROWS = t.n_rows
        self.COLUMN_GAP_MM = t.column_gap_mm
        self.ROW_GAP_MM = t.row_gap_mm
        self.COLUMNS = t.columns
        self.COLUMN_X_MM = t.column_x_mm


if not TEMPLATES:
    raise SystemExit(f"Aucun modèle de fiche valide trouvé dans {TEMPLATES_PATH}.")

active = ActiveConfig(DEFAULT_TEMPLATE_NAME)  # modèle sélectionné au démarrage (le premier du fichier)

# ----------------------------- Régions (référentiels de taxons) -----------------------------
# Le référentiel de taxons dépend de la région d'étude : une région = un fichier
# config/knowledge_taxons_<Région>.json (ex. knowledge_taxons_Metropole.json,
# knowledge_taxons_Antilles.json). Ajouter une région = déposer un nouveau fichier de ce type, sans
# toucher au code.
#
# Côté interface la colonne s'appelle toujours « Taxon » (clé 'taxon' dans modeles_fiches.json), mais en
# interne elle est traitée comme une entité propre à la région : « Taxon_Metropole », « Taxon_Antilles »...
# C'est ce nom d'entité qui sert de clé pour le décodage contraint, les exemples d'entraînement
# (training_data) et le classifieur entraîné (models/closed_set/<entité>.pt).
TAXONS_FILE_PREFIX = 'knowledge_taxons_'
DEFAULT_REGION = 'Metropole'

# Colonne (clé du modèle de fiche) -> préfixe de l'entité régionale correspondante.
REGIONAL_COLUMNS = {'taxon': 'Taxon'}

TAXONS_BY_REGION = {}   # région -> liste de {'nom', 'scientifique', 'code'}
for _p in sorted(CONFIG_DIR.glob(f'{TAXONS_FILE_PREFIX}*.json')):
    TAXONS_BY_REGION[_p.stem[len(TAXONS_FILE_PREFIX):]] = _load_json(_p, [])

if not TAXONS_BY_REGION:
    raise SystemExit(f"Aucun référentiel de taxons ({TAXONS_FILE_PREFIX}<Région>.json) trouvé dans {CONFIG_DIR}.")

# Région par défaut en premier, les autres par ordre alphabétique (ordre de la liste de choix de l'interface).
REGIONS = sorted(TAXONS_BY_REGION, key=lambda r: (r != DEFAULT_REGION, r))


def entity_name(column_key, region):
    """Nom interne de l'entité d'une colonne pour une région (ex. 'taxon', 'Antilles' -> 'Taxon_Antilles').
    Une colonne non régionale garde sa clé telle quelle."""
    prefix = REGIONAL_COLUMNS.get(column_key)
    return f'{prefix}_{region}' if prefix else column_key


def regional_entities():
    """{nom d'entité: (clé de colonne, région)} pour toutes les régions connues."""
    return {entity_name(col, reg): (col, reg) for col in REGIONAL_COLUMNS for reg in REGIONS}


class ActiveRegion:
    """Région actuellement sélectionnée dans l'application. Comme `active` pour le modèle de fiche,
    c'est un objet MUTABLE unique (voir `region` plus bas) : set_region() le met à jour en place."""

    def __init__(self, name):
        self.set_region(name)

    def set_region(self, name):
        if name not in TAXONS_BY_REGION:
            raise ValueError(f"Région inconnue : '{name}' (régions disponibles : {', '.join(REGIONS)})")
        self.name = name

    @property
    def taxons(self):
        return TAXONS_BY_REGION[self.name]

    def entity(self, column_key):
        return entity_name(column_key, self.name)


region = ActiveRegion(DEFAULT_REGION if DEFAULT_REGION in TAXONS_BY_REGION else REGIONS[0])


def resolve_key(column_key):
    """Clé « interne » d'une colonne pour la région active : 'taxon' -> 'Taxon_Metropole' ou
    'Taxon_Antilles' ; les autres colonnes ne changent pas."""
    return region.entity(column_key)


# ----------------------------- Colonnes à choix fermé -----------------------------
# Les valeurs possibles viennent de config/knowledge_donnees.json (une section par colonne) et
# config/knowledge_taxons_<Région>.json, plutôt que d'être recopiées en dur ici : modifier ces fichiers
# suffit à mettre à jour à la fois le menu déroulant de correction manuelle (app.py), le décodage
# contraint de l'OCR (app.py/handwriting_ocr.py) et le script d'entraînement des classifieurs
# dédiés (train_closed_set_classifiers.py). Indépendant du modèle de fiche sélectionné : ces
# colonnes ont le même vocabulaire quel que soit le modèle papier utilisé.
_DONNEES = _load_json(CONFIG_DIR / 'knowledge_donnees.json', {})

# Colonne (clé interne) -> section correspondante de knowledge_donnees.json.
_DONNEES_SECTIONS = {
    'sexe': 'sexe', 'taille_test': 'taille_testicules', 'taille_epi': 'taille_epididymes', 'tv': 'tv',
    'mam': 'mam', 'gest': 'gest', 'epiph': 'epiph', 'chin_spot': 'chin_spot', 'usure_dents': 'usure_dents',
    'glandes_taille': 'glandes_taille', 'glandes_coul': 'glandes_couleur', 'age_estime': 'age_estime',
}

CLOSED_SET_KEYS = {}    # clé colonne -> liste des codes valides (décodage contraint + menu déroulant)
CLOSED_SET_LABELS = {}  # clé colonne -> {code: "code — description"} pour un affichage plus lisible
for _key, _section in _DONNEES_SECTIONS.items():
    _vals = _DONNEES.get(_section) or {}
    if _vals:
        CLOSED_SET_KEYS[_key] = list(_vals.keys())
        CLOSED_SET_LABELS[_key] = {code: f'{code} — {desc}' for code, desc in _vals.items()}

# Taxons : une entité par région (Taxon_Metropole, Taxon_Antilles...), jamais une clé 'taxon' unique.
for _entity, (_col, _reg) in regional_entities().items():
    _taxons = TAXONS_BY_REGION[_reg]
    if _taxons:
        CLOSED_SET_KEYS[_entity] = list(dict.fromkeys(t['code'] for t in _taxons))
        CLOSED_SET_LABELS[_entity] = {t['code']: f"{t['code']} — {t['nom']}" for t in _taxons}


# ----------------------------- Observateurs (colonne Manip.) -----------------------------
# (config/observateurs.json) Initiales manuscrites -> nom complet de l'observateur. Plusieurs initiales peuvent pointer vers
# la même personne (ex. "MT" et "TM" pour la même personne) : la valeur finale enregistrée dans la
# case est toujours le nom complet, jamais les initiales.
OBSERVATEURS = _load_json(CONFIG_DIR / 'observateurs.json', {})
OBSERVATEUR_NAMES = sorted(set(OBSERVATEURS.values()))

# Manip. : classifieur fermé sur les noms des observateurs. Le modèle, s'il est entraîné,
# renvoie directement le nom complet ; le décodage conserve ensuite cette valeur.
if OBSERVATEURS:
    # Manip. : le modèle apprend les initiales manuscrites, pas les noms complets.
    # Les initiales sont récupérées depuis config/observateurs.json.
    manip_initials = set()
    for _key, _name in OBSERVATEURS.items():
        if _key:
            manip_initials.add(str(_key).strip().upper())
    CLOSED_SET_KEYS['manip'] = sorted(manip_initials)
    CLOSED_SET_LABELS['manip'] = {v: v for v in CLOSED_SET_KEYS['manip']}


# ----------------------------- Colonnes à "classificateur ouvert" -----------------------------
# AB et Poids : la reconnaissance de texte libre se trompe trop souvent sur ces deux colonnes.
# Plutôt qu'un ensemble fermé connu à l'avance (comme Sexe ou Taxon), on entraîne un classificateur
# dont le vocabulaire est constitué au fil des corrections des utilisateurs (train_closed_set_classifiers.py
# le déduit directement des valeurs observées dans training_data/{global,local}/manifest.csv). Il grossira avec
# l'usage — jusqu'à une centaine de valeurs environ.
OPEN_CLASSIFIER_KEYS = {'ab', 'poids'}
