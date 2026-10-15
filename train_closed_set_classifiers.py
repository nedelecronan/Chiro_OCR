"""
Entraîne un petit classifieur d'images pour :
- les colonnes à choix fermé (Sexe, Taille test., TV, Age estimé, Taxon...) — vocabulaire connu
  à l'avance, dans settings.CLOSED_SET_KEYS (issu de config/knowledge_donnees.json et de
  config/knowledge_taxons_<Région>.json) ;
- les colonnes à "classificateur ouvert" (AB, Poids — settings.OPEN_CLASSIFIER_KEYS) — la reconnaissance
  de texte libre s'y trompe trop souvent ; le vocabulaire est ici déduit des valeurs déjà validées/
  corrigées par les utilisateurs, et grossit avec l'usage (jusqu'à une centaine de valeurs environ).

Dans les deux cas, la donnée d'entraînement vient des cases validées ou corrigées manuellement dans
l'application (onglet Données : clic = validée, double-clic = corrigée). Le corpus est réparti en deux
sous-répertoires : training_data/global (données de référence) et training_data/local (alimenté par les
validations de l'application) ; les DEUX sont utilisés ici.

Le Taxon est entraîné séparément pour chaque région : entités « Taxon_Metropole », « Taxon_Antilles »...
(un classifieur models/closed_set/Taxon_<Région>.pt par région, car les vocabulaires diffèrent).

Ces colonnes n'ont qu'un nombre limité de valeurs manuscrites possibles. Plutôt que de faire deviner
du texte libre à TrOCR, on entraîne ici un petit classifieur d'image dédié à chaque colonne — un
problème beaucoup plus simple (proche d'un mini-MNIST) qui demande très peu de données et s'entraîne
en quelques minutes sur un simple CPU.

Une classe très majoritaire dans les données (ex. beaucoup de "PIPPIP" pour Taxon) est automatiquement
rééquilibrée pendant l'entraînement (échantillonnage pondéré + perte pondérée), pour qu'elle n'écrase
pas l'apprentissage des classes plus rares.

Prérequis (déjà dans requirements.txt) : torch, opencv-python, numpy.

Usage :
    python train_closed_set_classifiers.py                     # toutes les colonnes prêtes
    python train_closed_set_classifiers.py --key sexe          # une seule colonne
    python train_closed_set_classifiers.py --key ab            # colonne à classificateur ouvert
    python train_closed_set_classifiers.py --key taxon         # le Taxon de toutes les régions
    python train_closed_set_classifiers.py --region Antilles   # tout, mais Taxon des Antilles seulement
    python train_closed_set_classifiers.py --key Taxon_Antilles
    python train_closed_set_classifiers.py --min-examples 30   # seuil minimal avant d'entraîner
    python train_closed_set_classifiers.py --epochs 40

Les modèles entraînés sont enregistrés dans models/closed_set/<colonne>.pt et sont utilisés automatiquement par l'application : relisez le résumé de précision affiché en fin
d'entraînement avant de décider de les activer (voir remarque en bas de ce fichier).
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split

import training_store
from settings import CLOSED_SET_KEYS, OPEN_CLASSIFIER_KEYS, REGIONS, REGIONAL_COLUMNS, regional_entities

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models' / 'closed_set'
IMG_SIZE = 64  # chaque case est redimensionnée en carré IMG_SIZE x IMG_SIZE avant classification


def load_manifest():
    """Exemples de training_data/global ET training_data/local, regroupés par clé (colonne ou entité)."""
    training_store.migrate_legacy()   # ancienne structure (training_data/manifest.csv) -> local
    examples = training_store.load_examples()
    if not examples:
        raise SystemExit(
            f"Aucune donnée d'entraînement dans {training_store.GLOBAL_DIR} ni {training_store.LOCAL_DIR}.\n"
            "Validez ou corrigez quelques cases dans l'onglet Données de l'application d'abord."
        )
    by_key = defaultdict(list)
    for row in examples:
        by_key[row['key']].append(row)
    n_global = sum(1 for r in examples if r['origin'] == 'global')
    print(f"Corpus : {n_global} exemple(s) global + {len(examples) - n_global} exemple(s) local.\n")
    return by_key


class CellDataset(Dataset):
    def __init__(self, examples, classes):
        self.examples = examples
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        r = self.examples[i]
        img = cv2.imread(r['path'], cv2.IMREAD_COLOR)
        if img is None:
            # Image manquante/déplacée : image neutre plutôt que planter tout l'entraînement.
            img = np.full((IMG_SIZE, IMG_SIZE, 3), 235, dtype=np.uint8)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(img).permute(2, 0, 1)
        y = self.class_to_idx[r['value']]
        return x, y


class TinyCNN(nn.Module):
    # Volontairement minuscule : le vocabulaire est petit (2 à 6 classes par colonne),
    # inutile d'utiliser un réseau profond, on éviterait juste de sur-apprendre avec peu de données.
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 64 -> 32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_hour_models(examples, min_examples=20, epochs=25, batch_size=16, seed=0):
    """Entraîne 4 classifieurs digitaux à partir des exemples de la colonne heure.
    Chaque image doit contenir HH:MM (ou HHhMM). Les quatre zones sont extraites avant 64x64.
    """
    parsed=[]
    for r in examples:
        m=__import__('re').search(r'\b([01]?\d|2[0-3])\s*[:hH.]\s*([0-5]?\d)\b', r.get('value',''))
        if not m: continue
        hh=f"{int(m.group(1)):02d}"; mm=f"{int(m.group(2)):02d}"
        parsed.append((r, hh+mm))
    if len(parsed)<min_examples:
        print(f"[heure] seulement {len(parsed)} exemple(s) utilisable(s) (< {min_examples}), entraînement ignoré.")
        return
    class PosDataset(Dataset):
        def __init__(self, rows, pos, classes):
            self.rows=rows; self.pos=pos; self.classes=classes; self.map={c:i for i,c in enumerate(classes)}
        def __len__(self): return len(self.rows)
        def __getitem__(self,i):
            r,label=self.rows[i]
            img=cv2.imread(r['path'],cv2.IMREAD_COLOR)
            if img is None: img=np.full((IMG_SIZE,IMG_SIZE,3),235,np.uint8)
            h,w=img.shape[:2]
            boxes=[(0,int(w*.25)),(int(w*.25),int(w*.43)),(int(w*.57),int(w*.75)),(int(w*.75),w)]
            x1,x2=boxes[self.pos]; patch=img[:,x1:x2]
            patch=cv2.resize(patch,(IMG_SIZE,IMG_SIZE),interpolation=cv2.INTER_AREA)
            patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            return torch.from_numpy(patch).permute(2,0,1), self.map[label[self.pos]]
    for pos,(name,classes) in enumerate([
        ('heure_h10',['0','1','2']),('heure_h1',list('0123456789')),
        ('heure_m10',list('012345')),('heure_m1',list('0123456789'))]):
        torch.manual_seed(seed+pos)
        ds=PosDataset(parsed,pos,classes)
        n_val=max(1,int(.15*len(ds))); n_train=len(ds)-n_val
        tr,va=random_split(ds,[n_train,n_val],generator=torch.Generator().manual_seed(seed+pos))
        dl=DataLoader(tr,batch_size=min(batch_size,len(tr)),shuffle=True)
        vdl=DataLoader(va,batch_size=min(batch_size,len(va)))
        model=TinyCNN(len(classes)); opt=torch.optim.Adam(model.parameters(),lr=1e-3); lossf=nn.CrossEntropyLoss()
        best=0; best_state=None
        for ep in range(epochs):
            model.train()
            for x,y in dl:
                opt.zero_grad(); loss=lossf(model(x),y); loss.backward(); opt.step()
            model.eval(); cor=tot=0
            with torch.no_grad():
                for x,y in vdl:
                    pr=model(x).argmax(1); cor+=(pr==y).sum().item(); tot+=len(y)
            acc=cor/tot if tot else 0
            if acc>=best: best=acc; best_state={k:v.clone() for k,v in model.state_dict().items()}
        dest=MODELS_DIR/f'{name}.pt'
        torch.save({'state_dict':best_state,'classes':classes,'img_size':IMG_SIZE},dest)
        print(f"[heure] {name}: modèle enregistré, validation {best:.1%}")
    
def determine_classes(key, examples):
    """Renvoie (classes, mode) où mode vaut 'fermé' (vocabulaire connu à l'avance, ex. Sexe, Taxon)
    ou 'ouvert' (vocabulaire déduit des valeurs déjà observées, ex. AB, Poids — grossit avec l'usage)."""
    allowed = CLOSED_SET_KEYS.get(key)
    if allowed is not None:
        return sorted(set(allowed) | {''}), 'fermé'
    if key in OPEN_CLASSIFIER_KEYS:
        observed = sorted(set(r['value'] for r in examples) | {''})
        return observed, 'ouvert'
    return None, None


def train_one_column(key, examples, min_examples=20, epochs=25, batch_size=16, seed=0):
    classes, mode = determine_classes(key, examples)
    if classes is None:
        print(f"[{key}] ni une colonne à choix fermé connue (config.CLOSED_SET_KEYS), ni une colonne à "
              f"classificateur ouvert (config.OPEN_CLASSIFIER_KEYS), ignorée.")
        return

    counts = Counter(r['value'] for r in examples)
    origins = Counter(r.get('origin', '?') for r in examples)
    print(f"[{key}] mode {mode} — {len(examples)} exemple(s) collecté(s) {dict(origins)} — répartition : {dict(counts)}")

    usable = [r for r in examples if r['value'] in classes]
    ignored = len(examples) - len(usable)
    if ignored:
        print(f"[{key}] {ignored} exemple(s) ignoré(s) : valeur hors de la liste attendue {classes}.")

    if len(usable) < min_examples:
        print(f"[{key}] seulement {len(usable)} exemple(s) utilisable(s) (< --min-examples {min_examples}), "
              f"entraînement ignoré pour l'instant.\n")
        return
    present_classes = set(counts) & set(classes)
    if len(present_classes) < 2:
        print(f"[{key}] une seule valeur observée jusqu'ici ({present_classes}), "
              f"impossible d'entraîner un classifieur qui distingue plusieurs réponses. Ignoré.\n")
        return
    if mode == 'ouvert':
        print(f"[{key}] vocabulaire déduit des données ({len(classes)} valeur(s) distincte(s) observée(s)) : {classes}")

    torch.manual_seed(seed)
    ds = CellDataset(usable, classes)
    n_val = max(1, int(0.15 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    # Rééquilibrage des classes : une valeur très majoritaire dans les données (ex. beaucoup de
    # "PIPPIP" pour Taxon) sinon écrase l'apprentissage des autres classes. On utilise un
    # échantillonneur qui tire les exemples rares plus souvent pendant l'entraînement (chaque
    # époque voit un mélange plus équilibré des classes, sans dupliquer physiquement les données).
    # Note : combiner en plus une perte pondérée avec ce même échantillonneur sur-corrige et
    # inverse le déséquilibre (la classe autrefois majoritaire devient alors mal reconnue) —
    # un seul des deux leviers est utilisé ici, volontairement.
    train_examples = [usable[i] for i in train_ds.indices]
    train_counts = Counter(r['value'] for r in train_examples)
    sample_weights = [1.0 / train_counts[r['value']] for r in train_examples]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_examples), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), sampler=sampler)
    val_dl = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)))

    dominant = max(train_counts, key=train_counts.get)
    if train_counts[dominant] / len(train_examples) > 0.4 and len(train_counts) > 1:
        print(f"[{key}] classe majoritaire '{dominant}' : {train_counts[dominant]}/{len(train_examples)} "
              f"exemples d'entraînement — rééquilibrage actif (échantillonnage pondéré).")

    model = TinyCNN(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += len(y)
        acc = correct / total if total else 0.0
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"[{key}] époque {epoch:02d}/{epochs} — précision validation : {acc:.1%}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f'{key}.pt'
    torch.save({'state_dict': best_state, 'classes': classes, 'img_size': IMG_SIZE}, dest)
    print(f"[{key}] modèle enregistré -> {dest} (meilleure précision de validation : {best_acc:.1%})\n")



def get_closed_set_classes(key):
    """Return training classes. Manip uses initials from observateurs.json."""
    if key == "manip":
        try:
            from config import OBSERVATEURS
            return sorted(set(OBSERVATEURS.values()) | {""})
        except Exception:
            pass
    return list(CLOSED_SET_KEYS.get(key, []))

def select_keys(key=None, region=None):
    """Clés (colonnes / entités régionales) à entraîner. Les colonnes régionales sont désignées par leur
    entité (Taxon_Metropole, Taxon_Antilles...) ; 'taxon' est un raccourci pour toutes les régions."""
    entities = regional_entities()   # {entité: (colonne, région)}
    if key:
        if key in REGIONAL_COLUMNS:
            return sorted(e for e, (col, reg) in entities.items() if col == key and region in (None, reg))
        return [key]
    all_keys = set(CLOSED_SET_KEYS) | set(OPEN_CLASSIFIER_KEYS)
    if region:
        all_keys -= {e for e, (col, reg) in entities.items() if reg != region}
    return sorted(all_keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--key', help="Entraîner une seule colonne (ex. sexe). 'taxon' = le Taxon de toutes les régions "
                                  "(ou d'une seule avec --region) ; on peut aussi viser une entité (ex. Taxon_Antilles). "
                                  "Par défaut : toutes celles avec assez de données.")
    ap.add_argument('--region', choices=REGIONS, help="Limiter les colonnes régionales (Taxon) à cette région.")
    ap.add_argument('--min-examples', type=int, default=20, help="Nombre minimal d'exemples requis avant d'entraîner une colonne (défaut : 20).")
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()

    by_key = load_manifest()
    keys = select_keys(args.key, args.region)
    for key in keys:
        examples = by_key.get(key, [])
        if key == 'heure':
            if examples:
                train_hour_models(examples, min_examples=args.min_examples, epochs=args.epochs,
                                  batch_size=args.batch_size)
            else:
                print('[heure] aucun exemple collecté pour l\'instant.\n')
            continue
        if not examples:
            print(f"[{key}] aucun exemple collecté pour l'instant, ignorée.\n")
            continue
        train_one_column(key, examples, min_examples=args.min_examples, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()

# Les modèles présents dans models/closed_set/ sont chargés automatiquement par
# handwriting_ocr.py et prioritaires sur TrOCR pour les colonnes correspondantes.
# La classe vide est conservée comme classe valide afin d'éviter les valeurs inventées
# dans les cases sans écriture. Il est donc recommandé de conserver des exemples
# représentatifs de cases vides dans le jeu d'entraînement.
