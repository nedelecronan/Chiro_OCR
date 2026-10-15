"""Gestion du corpus d'entraînement (répertoire training_data/).

Le corpus est réparti en deux sous-répertoires de même structure :

    training_data/
        global/                 données « de référence », partagées (ex. versionnées avec le projet)
            manifest.csv
            images/<clé>/<uuid>.png
        local/                  données propres à ce poste, enrichies au fil des validations
            manifest.csv
            images/<clé>/<uuid>.png

- Valider ou corriger une case dans l'application écrit TOUJOURS dans « local » (record_example /
  remove_example) ; « global » n'est jamais modifié par l'application.
- L'entraînement des poids (.pt) lit « global » ET « local » (load_examples).

Les chemins d'image du manifeste sont relatifs au sous-répertoire qui contient ce manifeste, ce qui
permet de déposer/remplacer un jeu « global » complet sans rien réécrire.

Pour les colonnes régionales (Taxon), la clé enregistrée est le nom d'entité (« Taxon_Metropole »,
« Taxon_Antilles »...) : les exemples de chaque région restent donc séparés.
"""
import csv
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from settings import DEFAULT_REGION, REGIONAL_COLUMNS, entity_name

ROOT = Path(__file__).resolve().parent
TRAINING_DIR = ROOT / 'training_data'
GLOBAL_DIR = TRAINING_DIR / 'global'
LOCAL_DIR = TRAINING_DIR / 'local'
SOURCES = (('global', GLOBAL_DIR), ('local', LOCAL_DIR))   # ordre de lecture à l'entraînement
MANIFEST_NAME = 'manifest.csv'
MANIFEST_FIELDS = ['key', 'value', 'source', 'image', 'timestamp']


def manifest_path(directory):
    return Path(directory) / MANIFEST_NAME


def ensure_dirs():
    for _, d in SOURCES:
        (d / 'images').mkdir(parents=True, exist_ok=True)


def _read_manifest(directory):
    mp = manifest_path(directory)
    if not mp.exists():
        return []
    with open(mp, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=';'))


def _write_manifest(directory, rows):
    with open(manifest_path(directory), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter=';', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


# ----------------------------- lecture (entraînement) -----------------------------
def load_examples():
    """Tous les exemples de global + local. Chaque ligne du manifeste est enrichie de :
    'origin' ('global' ou 'local') et 'path' (chemin absolu de l'image)."""
    out = []
    for origin, directory in SOURCES:
        for row in _read_manifest(directory):
            row['origin'] = origin
            row['path'] = str(directory / row['image'])
            out.append(row)
    return out


# ----------------------------- écriture (application) : toujours dans « local » -----------------------------
def record_example(key, value, image_path, source):
    """Copie l'image de la case (telle que vue par l'OCR) dans training_data/local avec sa valeur validée
    par un humain. Renvoie le chemin relatif (à local/) de l'exemple enregistré, à fournir plus tard à
    remove_example si la validation est annulée ; None si l'image source n'existe pas."""
    src = Path(image_path)
    if not src.exists():
        return None
    ensure_dirs()
    rel = f'images/{key}/{uuid.uuid4().hex}.png'
    dest = LOCAL_DIR / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    rows = _read_manifest(LOCAL_DIR)
    rows.append({'key': key, 'value': value, 'source': source, 'image': rel,
                 'timestamp': datetime.now().isoformat(timespec='seconds')})
    _write_manifest(LOCAL_DIR, rows)
    return rel


def remove_example(rel_path):
    """Retire un exemple « local » (image ET ligne du manifeste). « global » n'est jamais touché."""
    img = LOCAL_DIR / rel_path
    if img.exists():
        img.unlink()
    rows = [r for r in _read_manifest(LOCAL_DIR) if r.get('image') != rel_path]
    if manifest_path(LOCAL_DIR).exists():
        _write_manifest(LOCAL_DIR, rows)


# ----------------------------- migration de l'ancienne structure -----------------------------
def migrate_legacy(log=print):
    """Ancienne structure : training_data/manifest.csv + training_data/images/<clé>/... Ces données
    deviennent le corpus « local ». L'ancienne colonne 'taxon' (qui n'existait que pour la
    Métropole) est renommée en entité régionale (Taxon_Metropole), images comprises.
    Renvoie le nombre d'exemples migrés (0 s'il n'y avait rien à faire)."""
    old_manifest = TRAINING_DIR / MANIFEST_NAME
    if not old_manifest.exists():
        return 0
    with open(old_manifest, encoding='utf-8-sig', newline='') as f:
        old_rows = list(csv.DictReader(f, delimiter=';'))
    ensure_dirs()
    migrated = _read_manifest(LOCAL_DIR)
    n = 0
    for r in old_rows:
        key = r['key']
        if key in REGIONAL_COLUMNS:
            key = entity_name(key, DEFAULT_REGION)
        old_img = TRAINING_DIR / r['image']
        rel = f"images/{key}/{old_img.name}"
        dest = LOCAL_DIR / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if old_img.exists():
            shutil.move(str(old_img), str(dest))
        migrated.append({'key': key, 'value': r['value'], 'source': r.get('source', ''),
                         'image': rel, 'timestamp': r.get('timestamp', '')})
        n += 1
    _write_manifest(LOCAL_DIR, migrated)
    old_manifest.rename(TRAINING_DIR / 'manifest.csv.migrated')
    old_images = TRAINING_DIR / 'images'
    if old_images.is_dir():
        for d in sorted(old_images.rglob('*'), reverse=True):   # supprime les dossiers désormais vides
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
        if old_images.is_dir() and not any(old_images.iterdir()):
            old_images.rmdir()
    log(f"Entraînement : {n} exemple(s) de l'ancienne structure migré(s) vers training_data/local.")
    return n
