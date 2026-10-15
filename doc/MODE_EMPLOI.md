# Mode d'emploi détaillé — Chiro OCR (V26)

! attention des mises à jour ont été effectué depuis dans la gestion des taxons (antilles/metropole) et dans la base de training !

Ce document explique **comment le code fonctionne**, de la photo de fiche papier jusqu'au CSV final,
et comment utiliser l'application au quotidien. Il complète les `README*.md` (qui retracent
l'historique des versions) sans les remplacer.

---

## 1. Vue d'ensemble

Le projet numérise des fiches papier de capture de chiroptères remplies à la main : chaque ligne
d'une fiche décrit un individu capturé (sexe, mesures, taxon, observateur, heure...). L'application :

1. redresse la photo de la fiche (perspective) ;
2. découpe l'image redressée en une grille de cases selon un **modèle de fiche** (une mise en page
   papier donnée) ;
3. lit chaque case (OCR manuscrit local, sans cloud) avec, pour certaines colonnes, un petit
   classifieur dédié entraîné sur les corrections précédentes ;
4. affiche le résultat dans un tableau où chaque case est colorée selon la confiance du décodage ;
5. permet de **valider** ou **corriger** chaque case d'un clic — ce qui alimente automatiquement un
   corpus d'entraînement (`training_data/`) pour améliorer les classifieurs au fil du temps ;
6. exporte le tout en CSV.

Tout tourne en local (pas d'appel réseau/cloud pendant le traitement).

---

## 2. Fichiers du projet

| Fichier | Rôle |
|---|---|
| `app.py` | Application graphique (Tkinter) : redressement, découpage, boucle de décodage, tableau interactif, export CSV. C'est le point d'entrée (`python app.py`). |
| `config.py` | Charge et expose toute la configuration : modèles de fiches (`modeles_fiches.json`), vocabulaires à choix fermé (`knowledge_donnees.json`, `knowledge_taxons.json`), observateurs (`observateurs.json`), et les groupes de mesures partagées. |
| `handwriting_ocr.py` | Le moteur OCR : charge TrOCR (générique) et les classifieurs fermés (`models/closed_set/*.pt`), et décide lequel utiliser pour chaque colonne. |
| `heure_utils.py` | Découpe une case "Heure Manip" en une zone "heures" et une zone "minutes" (voir §4.6.2). Utilisé à la fois à l'entraînement et à la lecture. |
| `train_closed_set_classifiers.py` | Script en ligne de commande qui entraîne les classifieurs `.pt` à partir des cases validées/corrigées dans l'application. |
| `modeles_fiches.json` | Décrit la géométrie de chaque modèle de fiche papier (colonnes, largeurs, nombre de lignes...). Voir §3. |
| `knowledge_donnees.json` | Vocabulaire des colonnes à choix fermé "simples" (Sexe, TV, Age estimé, ...). |
| `knowledge_taxons.json` | Liste des codes taxon valides. |
| `observateurs.json` | Table initiales manuscrites → nom complet de l'observateur (colonne Manip.). |
| `training_data/` | Corpus d'entraînement constitué automatiquement par les validations/corrections dans l'application (voir §7). Créé au premier lancement. |
| `sessions/` | Une sous-dossier par lot de photos chargé (photos copiées, images intermédiaires, CSV exporté). Créé au premier lancement. |
| `models/trocr-handwritten/` | Modèle TrOCR local (à télécharger séparément, voir `download_model.py` / `INSTALLATION_V7.md`). |
| `models/closed_set/*.pt` | Les petits classifieurs dédiés, générés par `train_closed_set_classifiers.py`. Absents au départ : l'application fonctionne sans eux (repli sur TrOCR), et devient plus précise à mesure qu'ils sont entraînés. |
| `viewer.py`, `decode.py`, `download_model.py` | Utilitaires annexes (visualisation rapide, décodage en ligne de commande, téléchargement du modèle TrOCR). |

---

## 3. Le modèle de fiche papier (`modeles_fiches.json`)

Toute la géométrie vient de ce fichier — aucune position n'est codée en dur dans `app.py`. Il peut
contenir plusieurs modèles (ex. `CACHI_2024`, `CACCHI_ancien`, `GMB_29`) : l'utilisateur choisit le
modèle actif dans le menu déroulant en haut de l'application.

Structure d'une entrée :

```json
"CACHI_2024": {
  "canonical_width": 2862,      // largeur (pixels) de l'image après redressement
  "canonical_height": 1839,     // hauteur (pixels) après redressement
  "form_width_mm": 286.2,       // largeur réelle de la fiche papier, en mm
  "form_height_mm": 183.9,
  "header_height_mm": 8.9,      // hauteur de la bande d'en-tête imprimée
  "row_height_mm": 7.0,         // hauteur d'une ligne de données
  "n_rows": 25,                 // nombre de lignes de la fiche
  "column_gap_mm": 0.1,         // marge à ignorer de part et d'autre d'un trait vertical imprimé
  "row_gap_mm": 0.1,            // idem pour les traits horizontaux
  "columns": [
    {"key": "n", "label": "N°", "width_mm": 4.0},
    {"key": "manip", "label": "Manip.", "width_mm": 12.0},
    {"key": "code_dispositif", "label": "Code dispositif", "width_mm": 12.3},
    ...
  ]
}
```
(Les commentaires `//` sont ajoutés ici pour expliquer ; le vrai fichier JSON n'en contient pas.)

**Point clé : la position d'une colonne ne dépend que de son ORDRE dans la liste `columns` et de sa
`width_mm`.** La colonne *i* commence exactement là où finit la colonne *i-1* (la première commence à
0). Il n'y a pas de position X indépendante à maintenir en plus des largeurs :

- **pour déplacer une colonne** (ex. inverser Tibia et Pouce), il suffit de réordonner les objets
  dans la liste `columns` ;
- **pour ajuster un espacement** entre deux colonnes (un espace physique sur le papier plus large
  qu'ailleurs), on augmente le `width_mm` de la colonne précédente d'autant.

`config.py` vérifie au chargement que la somme des `width_mm` correspond exactement à
`form_width_mm`, et que `header_height_mm + n_rows * row_height_mm` correspond à `form_height_mm` ;
un modèle qui ne respecte pas ça est rejeté avec un message d'erreur explicite (et l'application
continue avec les autres modèles valides).

`key` est l'identifiant interne stable de la colonne (utilisé partout dans le code et dans les CSV),
`label` est le texte affiché dans l'en-tête de l'interface. C'est aussi cette `key` qui détermine
quel vocabulaire/quel classifieur s'applique à la colonne (voir §5 et §4.6).

---

## 4. Le pipeline complet, étape par étape

Tout part de `App.decode_photo()` dans `app.py`, appelé une fois par photo chargée.

### 4.1 Redressement de la photo

`page_quad()` détecte automatiquement le contour de la fiche (contours + approximation
polygonale à 4 côtés, en choisissant le meilleur candidat par aire et par ratio largeur/hauteur
proche de celui attendu). Ce contour est présenté à l'utilisateur (`review_quad`) juste après le
chargement des photos : il peut faire glisser les 4 coins si la détection automatique s'est trompée,
avant de valider. `rectify()` applique ensuite une transformation de perspective vers une image
"canonique" de taille fixe (`CANONICAL_WIDTH` × `CANONICAL_HEIGHT`, définie par le modèle de fiche).

### 4.2 Découpage en grille

`fixed_boundaries()` calcule, à partir des largeurs de colonnes et hauteurs de lignes du modèle actif
(en mm), les positions en pixels (`xs`, `ys`) sur l'image canonique. `column_x_bounds()` /
`row_y_bounds()` rognent une petite marge (`column_gap_mm` / `row_gap_mm`) pour ne pas capturer les
traits imprimés. `crop_cell()` découpe la case avec une marge intérieure supplémentaire (8 % en
largeur, 12 % en hauteur) puis l'agrandit ×4 (interpolation cubique) : cet agrandissement aide
nettement l'OCR sur une petite case.

Le nombre de lignes réellement remplies sur la fiche est détecté automatiquement
(`count_used_rows`) en mesurant la densité d'encre de chaque ligne, avec tolérance à une ligne vide
isolée (deux lignes vides consécutives arrêtent la détection). On peut aussi forcer un nombre de
lignes fixe dans l'interface ("Lignes à décoder").

Pour chaque photo, `app.py` écrit dans `sessions/<session>/processed/<nom_photo>/` des images de
contrôle : `00_page_quad.jpg` (contour détecté), `01_canonical.jpg` (fiche redressée),
`02_decoupage.jpg` (grille superposée, numérotée), `geometry.json` (toutes les mesures utilisées), et
une image par case (`01_manip.png`, `01_heure.png`, ...) — ces images de case sont aussi celles
utilisées comme exemples d'entraînement (§7).

### 4.3 Détection des cases vides

Avant même d'appeler l'OCR, `ink_ratio_and_largest()` décide si une case contient réellement de
l'écriture. Chaque pixel est comparé à la couleur **médiane locale** de la case (pas à un seuil
absolu) : cela s'adapte à un fond blanc, bleuté ou rosé selon la colonne. Si le contraste est trop
faible (case uniforme) ou si la tache d'encre détectée est trop petite, la case est traitée comme
vide **sans jamais appeler l'OCR** — un modèle génératif comme TrOCR peut "halluciner" du texte sur
une case blanche, il ne faut donc pas lui laisser la décision.

### 4.4 Lecture d'une case (`ocr_read`)

Pour chaque case non vide, l'ordre de priorité est :

1. **Un classifieur fermé dédié existe pour cette colonne** (`o.has_closed_model(key)`, voir §4.6) →
   il est utilisé en priorité absolue. Il renvoie directement une classe de son vocabulaire (y
   compris explicitement "case vide").
2. **Colonne Manip. sans classifieur entraîné** → TrOCR contraint à un alphabet de lettres majuscules
   (2 à 3 caractères), puis résolution initiales → nom complet (§4.6.1).
3. **Colonne Heure sans classifieur entraîné** → TrOCR contraint au jeu de caractères
   `0123456789:hH.` (3 à 5 caractères).
4. **Autres colonnes** → TrOCR, avec deux formes de contrainte possibles :
   - `numeric=True` pour les colonnes numériques (`NUMERIC_OCR_KEYS`) : seuls les tokens composés de
     chiffres/`,`/`.`/`-`/espace sont autorisables pendant la génération ;
   - `closed_set=CLOSED_SET_KEYS.get(key)` pour les colonnes à vocabulaire connu (Sexe, Taxon,
     Age estimé, ...) : la génération est **forcée** à suivre un "trie" (arbre de préfixes) construit
     à partir des valeurs autorisées, donc TrOCR ne peut physiquement pas produire une valeur hors
     de ce vocabulaire (y compris la valeur vide).

### 4.5 Validation métier (`constrained`)

Le texte brut renvoyé par l'OCR (`raw`) passe ensuite dans `constrained()`, qui applique des règles
propres à chaque colonne (par son `label` d'affichage, ex. `"Sexe"`, `"Manip."`, `"Heure Manip"`) :

- `"` (guillemet) → reprend la valeur de la ligne précédente pour cette colonne ("idem") ;
- valeur vide → case vide, confiance forte ;
- `Sexe` → règles textuelles (F/M/symboles) ;
- `Manip.` → nettoyage en 2-3 lettres majuscules ;
- `Heure Manip` → expression régulière `HH[:hH.]MM` avec validation des plages horaires ;
- colonnes numériques (`NUM_RULES`) → extraction du nombre, vérification qu'il tombe dans une plage
  plausible (ex. AB entre 27 et 120 mm), avec un format attendu par colonne (`NUM_PATTERNS` : AB et
  Tibia doivent comporter une décimale) ;
- colonnes à choix fermé (`CLOSED_SET_KEYS`) → comparaison textuelle avec le vocabulaire connu ;
- sinon → texte libre tel quel, confiance faible.

Chaque règle renvoie `(valeur, confiance, niveau, source)`. Le `niveau` (`fort`/`moyen`/`faible`)
détermine la couleur de fond de la case dans le tableau (§6.3).

**Cas particulier** : quand `ocr_read` a utilisé un classifieur fermé, `decode_photo` **écrase**
ensuite le niveau de confiance calculé par `constrained()` avec celui, plus fiable, renvoyé par le
classifieur (probabilité softmax de la classe prédite) — sauf pour le "idem" et la case vide, qui
restent gérés par la même logique que d'habitude.

### 4.6 Cas particuliers

#### 4.6.1 Manip. : initiales manuscrites → nom complet

`observateurs.json` fait correspondre des initiales (`MT`, `TM`, `RN`, ...) à un nom complet
(plusieurs initiales peuvent pointer vers la même personne). Le classifieur `manip.pt` (s'il existe)
et le repli TrOCR lisent tous les deux des **initiales**, jamais un nom complet. `decode_photo`
conserve ces initiales à part (`row['manip__initials']`) avant de les résoudre en nom complet
(`resolve_observateur`, avec un rapprochement flou si les initiales exactes ne sont pas trouvées) —
c'est le nom complet qui est affiché et exporté, mais ce sont les initiales qui sont réutilisées pour
le "idem" d'une ligne à l'autre et pour l'entraînement (§7).

#### 4.6.2 Heure Manip : segmentation en deux zones

Ancienne méthode (abandonnée) : découper la case en 4 boîtes de largeur **fixe** (dizaine/unité
d'heure, dizaine/unité de minute), en supposant une heure toujours écrite sur exactement 2 chiffres à
une position figée. Cela échoue dès qu'une heure n'a qu'un seul chiffre ("9h55"), que le séparateur
(`h`, `:`, `.`) change de forme ou de position, ou que les deux chiffres des minutes sont tracés d'un
seul geste continu.

Nouvelle méthode (`heure_utils.split_hour_minute`), appliquée **à l'identique** à l'entraînement et à
la lecture :

1. Binarisation encre/fond auto-calibrée (même principe qu'au §4.3).
2. Regroupement en composantes connexes, fusionnées quand leurs empreintes horizontales se
   chevauchent (un `:` produit deux points superposés verticalement mais au même endroit en X).
3. Les minutes sont toujours exactement 2 chiffres, qu'ils soient écrits séparément (2 groupes) ou
   d'un seul trait (1 groupe nettement plus large que les autres) : on compare la largeur du dernier
   groupe à la largeur typique (médiane) des groupes précédents pour choisir entre les deux cas. Tout
   ce qui précède constitue la zone "heures" — un éventuel reste du séparateur qui s'y retrouve
   mélangé n'est pas un problème, car cette zone n'est pas lue chiffre par chiffre : elle est classée
   **comme une image entière**.
4. Chaque zone est classée par un petit CNN dédié : `heure_heures.pt` (vocabulaire "00" à "23") et
   `heure_minutes.pt` (vocabulaire "00" à "59"). La confiance finale est la moyenne des deux.

Sans ces deux modèles, `ocr_read` se replie sur TrOCR contraint (§4.4, point 3).

#### 4.6.3 Mesures groupées : `mesures_decimales` et `mesure_doigts`

Poids, Tibia, AB, Pouce, Pied et CM3 partagent un seul classifieur, `mesures_decimales.pt` ; D3 et D5
partagent `mesure_doigts.pt` (défini dans `config.MEASURE_GROUPS`). Une même valeur manuscrite (ex.
"30,4") revient souvent sur plusieurs de ces colonnes : les regrouper démultiplie le nombre
d'exemples d'entraînement par classe par rapport à un modèle par colonne. `config.COLUMN_TO_
MEASURE_MODEL` fait la correspondance colonne → nom du modèle partagé ; c'est ce que
`handwriting_ocr.has_closed_model()` / `closed_set_predict()` utilisent pour retrouver le bon `.pt`
quelle que soit la colonne demandée. Voir §7 et §8 pour la façon dont l'entraînement regroupe les
données tout en les gardant séparées par colonne dans `training_data/`.

---

## 5. Les référentiels de connaissances

Trois fichiers JSON, chargés par `config.py`, définissent tout le vocabulaire "à choix fermé" —
modifier ces fichiers suffit à mettre à jour en même temps le menu déroulant de correction, le
décodage contraint de l'OCR, et le script d'entraînement, sans toucher au code :

- **`knowledge_donnees.json`** : une section par colonne "simple" (`sexe`, `taille_testicules`,
  `tv`, `mam`, `gest`, `epiph`, `chin_spot`, `usure_dents`, `glandes_taille`, `glandes_couleur`,
  `age_estime`), chacune un dictionnaire `{code: description}`. `config._DONNEES_SECTIONS` fait la
  correspondance entre la clé interne d'une colonne (ex. `taille_test`) et son nom de section dans ce
  fichier (ex. `taille_testicules`).
- **`knowledge_taxons.json`** : une liste d'objets `{code, nom, scientifique}` pour la colonne Taxon.
- **`observateurs.json`** : `{initiales: "Nom complet"}` pour la colonne Manip. (§4.6.1).

À partir de ces fichiers, `config.py` construit :
- `CLOSED_SET_KEYS` : `{clé_colonne: [valeurs valides]}` — utilisé pour le décodage contraint
  (`ocr_read`/`constrained`), le menu déroulant de correction (`edit_cell_grid`), et l'entraînement ;
- `CLOSED_SET_LABELS` : `{clé_colonne: {code: "code — description"}}` pour un affichage plus lisible
  dans le menu déroulant (seul le code est conservé comme valeur de la case) ;
- `OBSERVATEUR_NAMES` et `OBSERVATEUR_INITIALS_BY_NAME` (nom → initiales possibles, utilisé lors
  d'une correction manuelle de Manip., §4.6.1).

Dans l'application, l'onglet **Référentiels** permet d'éditer deux tables auxiliaires
(`referentiels/Manip.csv`, `referentiels/taxons.csv`) utilisées par un ancien mécanisme de
rapprochement flou (`match_ref`) — ce mécanisme n'est plus utilisé pour le décodage (remplacé par le
décodage contraint du §4.4), il reste disponible mais n'a plus d'effet sur la lecture des cases.

---

## 6. L'interface graphique : guide d'utilisation

### 6.1 Lancer l'application

```bash
python app.py
```
(ou double-cliquer `lancer_chiro_ocr.bat` sous Windows). Voir `INSTALLATION_V7.md` pour l'installation
de l'environnement (Python, dépendances, modèle TrOCR local).

### 6.2 Charger et décoder des photos

1. **Modèle de fiche** : choisir dans le menu déroulant en haut le modèle correspondant aux photos à
   traiter (`CACHI_2024`, `CACCHI_ancien`, `GMB_29`...). Changer de modèle efface les données déjà
   décodées non enregistrées (une confirmation est demandée).
2. **📷 Charger des JPG** : sélectionner une ou plusieurs photos. Une nouvelle session est créée
   (`sessions/AAAAMMJJ_HHMMSS/`), les photos y sont copiées. Pour chaque photo, une fenêtre
   s'ouvre avec le contour détecté automatiquement (en rouge) : faire glisser les coins si besoin,
   puis "✓ Valider ce contour" (ou "↺ Redétecter" pour relancer la détection automatique).
3. **Lignes à décoder** : laisser à `0` pour une détection automatique du nombre de lignes remplies,
   ou forcer un nombre précis.
4. **▶ Décoder** : lance le pipeline complet (§4) sur toutes les photos chargées et remplit le
   tableau de l'onglet **Données**.

### 6.3 Le tableau de données

Chaque case est colorée selon la confiance du décodage :
- **blanc** : confiance forte ;
- **jaune pâle** : confiance moyenne ;
- **rose** : confiance faible — à vérifier en priorité ;
- **vert clair** : case cliquée pour valider telle quelle ;
- **vert foncé** : case corrigée manuellement.

Interactions :
- **clic simple** sur une case → la valide telle quelle (elle devient vert clair) et enregistre un
  exemple d'entraînement (§7). Re-cliquer sur une case déjà validée/corrigée **annule** cette
  validation (retire l'exemple d'entraînement correspondant).
- **double-clic** sur une case → ouvre une boîte de correction. Pour une colonne à vocabulaire connu
  (Sexe, Taxon, Manip., ...), c'est un menu déroulant fermé (impossible de saisir une valeur
  invalide) ; sinon, une saisie libre au clavier. Valider la correction colore la case en vert foncé
  et enregistre un exemple d'entraînement avec la valeur corrigée.
- la colonne `N°` n'est ni cliquable ni éditable (numéro de ligne généré, pas manuscrit).

La barre de statut en bas indique le nombre de lignes, de cases traitées (validées + corrigées), et
combien de cases à confiance faible restent à vérifier.

### 6.4 Exporter

**💾 Enregistrer CSV** écrit `sessions/<session>/donnees_decodees.csv` (une colonne par `key` du
modèle de fiche actif, dans l'ordre du JSON).

### 6.5 Autres onglets

- **Historique sessions** : liste des lots de photos déjà traités.
- **Référentiels** : édition de `referentiels/Manip.csv` et `referentiels/taxons.csv` (voir la
  remarque du §5 sur leur rôle actuel limité).
- **Journal** : messages techniques (erreurs OCR, chargement du modèle, changements de modèle de
  fiche...) — à consulter en cas de comportement inattendu.

---

## 7. L'apprentissage continu (`training_data/`)

Chaque fois qu'une case est validée (clic) ou corrigée (double-clic), `record_training_example()` :

1. copie l'image de la case (`training_data/images/<clé_colonne>/<uuid>.png`) — c'est exactement le
   même recadrage que celui vu par l'OCR (§4.2), pas la photo entière ;
2. ajoute une ligne à `training_data/manifest.csv` (séparateur `;`) :

   | colonne | contenu |
   |---|---|
   | `key` | clé de la colonne (ex. `sexe`, `poids`, `manip`, `heure`) |
   | `value` | la valeur validée/corrigée — pour `manip`, ce sont les **initiales**, pas le nom complet (§4.6.1) |
   | `source` | `validated` ou `corrected` |
   | `image` | chemin relatif de l'image copiée |
   | `timestamp` | date/heure de l'enregistrement |

Annuler une validation (re-cliquer sur une case déjà colorée) ou corriger une case déjà
validée/corrigée retire l'exemple précédent (`remove_training_example`) avant d'en enregistrer un
nouveau : il n'y a jamais qu'un seul exemple d'entraînement par case à un instant donné.

Ce corpus s'accumule sur toutes les fiches traitées, indépendamment des sessions (qui peuvent être
nettoyées sans perdre l'entraînement). C'est la matière première de `train_closed_set_classifiers.py`
(§8).

---

## 8. Entraîner les classifieurs (`train_closed_set_classifiers.py`)

```bash
python train_closed_set_classifiers.py                        # tout ce qui a assez de données
python train_closed_set_classifiers.py --key sexe              # une seule colonne à choix fermé
python train_closed_set_classifiers.py --key poids              # entraîne 'mesures_decimales' (groupe)
python train_closed_set_classifiers.py --key mesures_decimales  # idem, par le nom du groupe
python train_closed_set_classifiers.py --key mesure_doigts      # D3/D5
python train_closed_set_classifiers.py --key heure               # heure_heures + heure_minutes
python train_closed_set_classifiers.py --min-examples 30 --epochs 40
```

Ce que fait le script pour une colonne "simple" (ex. `sexe`) :

1. lit `training_data/manifest.csv`, regroupe les exemples par `key` ;
2. détermine le vocabulaire attendu (`config.CLOSED_SET_KEYS[key]`, plus la classe vide) ;
3. ignore les exemples dont la valeur n'est pas dans ce vocabulaire (affiché dans les logs) ;
4. si moins de `--min-examples` exemples utilisables, ou une seule valeur observée, l'entraînement
   est **ignoré pour l'instant** (message explicite, pas une erreur) ;
5. sinon, entraîne un petit CNN (`TinyCNN`, 3 blocs conv+pool) sur les images 64×64, avec un
   échantillonnage pondéré par classe pour compenser un éventuel déséquilibre (une valeur très
   majoritaire dans les données) ;
6. affiche la précision de validation à chaque époque, et sauvegarde le meilleur état dans
   `models/closed_set/<key>.pt`.

Pour les **mesures groupées** (§4.6.3), `train_measure_group()` fait la même chose mais après avoir
fusionné les exemples de toutes les colonnes du groupe (le vocabulaire est déduit dynamiquement des
valeurs observées, puisqu'il n'y a pas de liste fixe pour un poids ou une longueur).

Pour l'**heure** (§4.6.2), `train_hour_models()` ré-applique la segmentation `heure_utils` à chaque
image de case validée pour produire deux crops étiquetés (heures, minutes), puis entraîne les deux
classifieurs `heure_heures.pt` / `heure_minutes.pt` séparément.

**Quand relancer l'entraînement ?** Il n'y a pas d'automatisme : relancer ce script manuellement de
temps en temps (par exemple après une session de validation/correction significative), puis relancer
l'application pour qu'elle charge les nouveaux `.pt`. Il est recommandé de regarder la précision de
validation affichée en fin d'entraînement avant de faire confiance à un nouveau modèle sur des
données réelles — un score bas signale qu'il manque encore d'exemples pour cette colonne.

---

## 9. Ajouter ou modifier un modèle de fiche

1. Ajouter une entrée dans `modeles_fiches.json` (voir §3) avec toutes les colonnes dans leur ordre
   physique réel sur le papier, en mm.
2. Vérifier que la somme des `width_mm` correspond exactement à `form_width_mm`, et que
   `header_height_mm + n_rows * row_height_mm` correspond à `form_height_mm` — sinon `config.py`
   rejette le modèle au démarrage avec un message précis.
3. Relancer l'application : le nouveau modèle apparaît dans le menu déroulant.
4. Charger une photo de test, décoder, et vérifier `sessions/.../processed/.../02_decoupage.jpg`
   (grille superposée, numérotée) pour confirmer que chaque colonne tombe au bon endroit.

Les colonnes utilisant un vocabulaire à choix fermé ou un classifieur partagé sont reconnues par leur
**clé** (`key`), qui doit correspondre à celles attendues ailleurs dans le code : voir
`config._DONNEES_SECTIONS`, `config.MEASURE_GROUPS`, et la clé `manip`/`heure`/`taxon`. Une colonne
avec une clé inconnue de ces listes est traitée comme du texte libre (OCR non contraint).

---

## 10. Dépannage / FAQ

**"Aucun texte n'a pu être lu" après décodage.** Ni TrOCR ni les classifieurs fermés ne fonctionnent.
Regarder l'onglet Journal : le message le plus courant est un modèle TrOCR local absent
(`models/trocr-handwritten/`, voir `download_model.py`) ou une dépendance manquante
(`sentencepiece`, `tokenizers`...).

**Une colonne "à choix fermé" n'accepte pas une valeur légitime.** Vérifier
`knowledge_donnees.json` / `knowledge_taxons.json` / `observateurs.json` : la valeur doit y figurer.
Ces fichiers pilotent aussi bien le menu déroulant que le décodage contraint.

**Un classifieur semble "toujours donner la même réponse".** C'est le symptôme historique du bug
Manip. corrigé en V26 (§4.6.1, README_V26.md) : vérifier que la valeur enregistrée dans
`training_data/manifest.csv` pour cette colonne correspond bien au vocabulaire attendu
(`config.CLOSED_SET_KEYS[key]` pour une colonne à choix fermé, ou une valeur numérique cohérente pour
une mesure groupée). Un classifieur entraîné sur des étiquettes qui ne correspondent à aucune classe
réelle n'apprend rien d'utile.

**Une case Heure Manip est mal découpée (heure/minute mélangées).** Regarder l'image de la case dans
`sessions/.../processed/.../<ligne>_heure.png` : si l'écriture est très dense ou le séparateur
inhabituel, la segmentation (§4.6.2) peut se tromper. Corriger la case manuellement (elle sert alors
d'exemple d'entraînement) plutôt que de forcer une valeur incohérente.

**Changer de modèle de fiche efface mes données.** C'est volontaire (la géométrie change) : bien
enregistrer le CSV avant de changer de modèle si les données actuelles doivent être conservées.
