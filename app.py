import csv, json, re, shutil, subprocess, sys, traceback
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import cv2
import numpy as np
from PIL import Image as PILImage, ImageTk

from settings import (active as cfg, region, resolve_key, REGIONS, BLANK_STD_THRESHOLD, BLANK_INK_RATIO, BLANK_MIN_AREA_RATIO,
                      CLOSED_SET_KEYS, CLOSED_SET_LABELS, TEMPLATES, OBSERVATEURS, OBSERVATEUR_NAMES)
from handwriting_ocr import LocalHandwritingOCR
import training_store

ROOT = Path(__file__).resolve().parent
REF_DIR = ROOT / 'referentiels'
SESSIONS_DIR = ROOT / 'sessions'
MODEL_DIR = ROOT / 'models' / 'trocr-handwritten'
SESSIONS_DIR.mkdir(exist_ok=True)
REF_DIR.mkdir(exist_ok=True)
# Corpus d'entraînement : training_data/global (données de référence, jamais modifiées par l'application)
# et training_data/local (constitué au fil des validations/corrections manuelles dans l'onglet Données).
# Indépendant des sessions (qui peuvent être nettoyées) : il s'accumule sur toutes les fiches traitées.
# Voir training_store.py. L'ancienne structure (training_data/manifest.csv) est migrée vers « local ».
training_store.ensure_dirs()
training_store.migrate_legacy()

# ----------------------------- image geometry -----------------------------
def load_img(path):
    img=cv2.imread(str(path))
    if img is None: raise ValueError(f'Impossible de lire {path}')
    return img

def order_quad(p):
    p=np.asarray(p,dtype=np.float32)
    s=p.sum(1); d=np.diff(p,axis=1).ravel()
    return np.array([p[np.argmin(s)],p[np.argmin(d)],p[np.argmax(s)],p[np.argmax(d)]],np.float32)

def page_quad(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    g=cv2.GaussianBlur(g,(5,5),0)
    edges=cv2.Canny(g,40,130)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(25,25)))
    contours,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    H,W=g.shape; target=cfg.CANONICAL_WIDTH/cfg.CANONICAL_HEIGHT
    best=None
    for c in contours:
        area=cv2.contourArea(c)
        if area < .55*H*W: continue
        peri=cv2.arcLength(c,True)
        q=cv2.approxPolyDP(c,.02*peri,True)
        if len(q)!=4: continue
        pts=order_quad(q[:,0,:])
        qw=max(np.linalg.norm(pts[1]-pts[0]),np.linalg.norm(pts[2]-pts[3]))
        qh=max(np.linalg.norm(pts[3]-pts[0]),np.linalg.norm(pts[2]-pts[1]))
        ratio=qw/max(qh,1)
        # Do not accept a wildly wrong trapezoid/side object.
        ratio_penalty=abs(np.log(max(ratio,.01)/target))
        score=area/(1+2.0*ratio_penalty)
        if best is None or score>best[0]: best=(score,pts)
    if best is None:
        return np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]],np.float32),False
    return best[1],True

def rectify(img,quad=None):
    if quad is not None:
        q,ok=order_quad(quad),True
    else:
        q,ok=page_quad(img)
    dst=np.array([[0,0],[cfg.CANONICAL_WIDTH-1,0],[cfg.CANONICAL_WIDTH-1,cfg.CANONICAL_HEIGHT-1],[0,cfg.CANONICAL_HEIGHT-1]],np.float32)
    M=cv2.getPerspectiveTransform(q,dst)
    out=cv2.warpPerspective(img,M,(cfg.CANONICAL_WIDTH,cfg.CANONICAL_HEIGHT),borderMode=cv2.BORDER_REPLICATE)
    # Canonical form is landscape. Only rotate if the resulting page is portrait.
    if out.shape[0]>out.shape[1]:
        out=cv2.rotate(out,cv2.ROTATE_90_CLOCKWISE)
    return out,q,ok

def fixed_boundaries():
    # X peut être défini indépendamment pour chaque colonne via x_mm dans modeles_fiches.json.
    # Cela permet de décaler une colonne sans devoir modifier toutes les suivantes.
    x_mm = getattr(cfg, 'COLUMN_X_MM', None)
    xs=[]
    for i,c in enumerate(cfg.COLUMNS):
        left = x_mm[i] if x_mm is not None else sum(cc.width_mm for cc in cfg.COLUMNS[:i])
        xs.append(int(round(left/cfg.FORM_WIDTH_MM*(cfg.CANONICAL_WIDTH-1))))
    right_mm = (x_mm[-1] + cfg.COLUMNS[-1].width_mm) if x_mm is not None else cfg.FORM_WIDTH_MM
    xs.append(int(round(right_mm/cfg.FORM_WIDTH_MM*(cfg.CANONICAL_WIDTH-1))))
    xs[0]=0
    if abs(right_mm-cfg.FORM_WIDTH_MM)<1e-6: xs[-1]=cfg.CANONICAL_WIDTH-1
    ys=[0]
    header=cfg.HEADER_HEIGHT_MM/cfg.FORM_HEIGHT_MM*(cfg.CANONICAL_HEIGHT-1)
    row=cfg.ROW_HEIGHT_MM/cfg.FORM_HEIGHT_MM*(cfg.CANONICAL_HEIGHT-1)
    ys.append(int(round(header)))
    for i in range(1,cfg.N_ROWS): ys.append(int(round(header+i*row)))
    ys.append(cfg.CANONICAL_HEIGHT-1)
    return xs,ys

def row_y_bounds(ys,r):
    # ys[0]:ys[1] is the header band; data row r (0-based) is band r+1, i.e. ys[r+1]:ys[r+2].
    # Trims ROW_GAP_MM (half on each shared edge) so the printed horizontal separator isn't captured.
    # The very top of the page (ys[0]) and the very bottom (ys[-1]) are table edges, not shared lines, so they stay untouched.
    band=r+1
    gap_px=cfg.ROW_GAP_MM/cfg.FORM_HEIGHT_MM*(cfg.CANONICAL_HEIGHT-1)
    y1=ys[band]+(gap_px/2 if band>0 else 0)
    y2=ys[band+1]-(gap_px/2 if band+1<len(ys)-1 else 0)
    return int(round(y1)),int(round(y2))

def column_x_bounds(xs,ci):
    # Trims COLUMN_GAP_MM (half on each shared edge) so the printed vertical separator isn't captured.
    gap_px=cfg.COLUMN_GAP_MM/cfg.FORM_WIDTH_MM*(cfg.CANONICAL_WIDTH-1)
    x1=xs[ci]+(gap_px/2 if ci>0 else 0)
    x2=xs[ci+1]-(gap_px/2 if ci<len(xs)-2 else 0)
    return int(round(x1)),int(round(x2))

def ink_ratio_and_largest(bgr):
    # Works on the colour patch, not grayscale: converting to grayscale first washes out
    # the contrast between dark ink and a pastel (blue/pink) printed background, which was
    # causing real handwriting on tinted columns to be misread as blank.
    # Instead of an absolute darkness/threshold, every pixel is compared to the cell's own
    # background colour (its per-channel median, since background pixels dominate any cell),
    # so it self-calibrates to whatever the local background colour is (white, blue or pink).
    if bgr.size==0:return 0.0,0
    if bgr.ndim==2: bgr=cv2.cvtColor(bgr,cv2.COLOR_GRAY2BGR)
    flat=bgr.reshape(-1,3).astype(np.float32)
    bg=np.median(flat,axis=0)
    diff=np.linalg.norm(bgr.astype(np.float32)-bg,axis=2)
    diff=np.clip(diff,0,255).astype(np.uint8)
    # Guard against Otsu forcing a ~bimodal split on an essentially uniform (blank) patch,
    # which was the main source of blank cells/rows being wrongly read as having content.
    if float(np.std(diff))<BLANK_STD_THRESHOLD:
        return 0.0,0
    bw=cv2.threshold(diff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    bw=cv2.morphologyEx(bw,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    ink=float(np.mean(bw>0))
    nlabels,labels,stats,_=cv2.connectedComponentsWithStats(bw,8)
    areas=[int(a) for a in stats[1:,cv2.CC_STAT_AREA]]
    largest=max(areas) if areas else 0
    return ink,largest

def row_ink_ratio(can,xs,ys,r):
    # Ink density of a row (header excluded), columns after 'n°' only (that column is printed/generated, not handwritten).
    x1,x2=xs[1],xs[-1]; y1,y2=row_y_bounds(ys,r)
    band=can[y1:y2,x1:x2]
    if band.size==0:return 0.0
    band=cv2.GaussianBlur(band,(3,3),0)
    ink,_=ink_ratio_and_largest(band)
    return ink

def count_used_rows(can,xs,ys,n_rows=None,ink_threshold=0.015,max_empty_streak=2):
    # n_rows par défaut = cfg.N_ROWS du modèle de fiche ACTIF au moment de l'appel, pas au chargement
    # du module (un paramètre par défaut ne serait évalué qu'une fois, ce qui figerait l'ancien modèle).
    if n_rows is None: n_rows=cfg.N_ROWS
    # Stops after two consecutive rows without ink, to tolerate one accidentally-skipped line.
    used=0; empty_streak=0
    for r in range(n_rows):
        if row_ink_ratio(can,xs,ys,r)>ink_threshold:
            used=r+1; empty_streak=0
        else:
            empty_streak+=1
            if empty_streak>=max_empty_streak:break
    return max(used,1)

def draw_grid(img,xs,ys,path):
    d=img.copy()
    for i,x in enumerate(xs):
        cv2.line(d,(x,0),(x,d.shape[0]-1),(0,0,255),1)
        if i<len(xs)-1: cv2.putText(d,str(i+1),(x+3,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1,cv2.LINE_AA)
    for i,y in enumerate(ys):
        cv2.line(d,(0,y),(d.shape[1]-1,y),(255,0,0),1)
        if i<len(ys)-1: cv2.putText(d,str(i+1),(3,y+16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,0,0),1,cv2.LINE_AA)
    cv2.imwrite(str(path),d)

def crop_cell(img,x1,x2,y1,y2):
    # Keep a generous interior margin to exclude the printed grid.
    mx=max(4,int((x2-x1)*.08)); my=max(5,int((y2-y1)*.12))
    c=img[y1+my:y2-my,x1+mx:x2-mx]
    if c.size==0:return img[0:1,0:1]
    return cv2.resize(c,None,fx=4,fy=4,interpolation=cv2.INTER_CUBIC)

# ----------------------------- references -----------------------------
def ensure_refs():
    for name in ['Manip.csv','taxons.csv']:
        p=REF_DIR/name
        if not p.exists():
            src=ROOT/name
            if src.exists(): shutil.copy2(src,p)

def read_csv(path):
    if not path.exists(): return []
    with open(path,'r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,delimiter=';'))

def write_csv(path, rows, fields):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter=';');w.writeheader();w.writerows(rows)

# ----------------------------- OCR constraints -----------------------------
NUM_RULES={
 'Poids':(3,60), 'AB':(27,120), 'D5':(30,120), 'D3':(30,120),
 'Tibia':(10,30), 'CM3':(5,15), 'Queue':(5,60), 'Code dispositif':(1,8)
}
# Format attendu pour chaque mesure : AB et Tibia doivent obligatoirement comporter une décimale
# (ex. 30,4) ; Poids en a généralement une ou deux mais peut rester entier ; les autres restent
# permissives (entier ou décimal, comme avant).
NUM_PATTERNS={
 'AB':r'\d+[.,]\d+', 'Tibia':r'\d+[.,]\d+', 'Poids':r'\d+(?:[.,]\d{1,2})?',
}
NUM_PATTERN_DEFAULT=r'\d+(?:[.,]\d+)?'
NUMERIC_OCR_KEYS={'ab','d5','d3','cm3','poids','queue','tibia','code_dispositif'}
# Colonnes numériques où une correction manuelle tapée avec un "." doit être convertie en ",".
DECIMAL_KEYS={'poids','ab','d5','d3','tibia','cm3','queue'}
# CLOSED_SET_KEYS/CLOSED_SET_LABELS viennent de settings.py (chargés depuis config/knowledge_donnees.json
# et config/knowledge_taxons_<Région>.json ; le Taxon y est une entité par région, cf. resolve_key), partagés avec le décodage contraint côté OCR et train_closed_set_classifiers.py.

def norm(s): return re.sub(r'[^a-z0-9+?.,:-]+',' ',str(s).lower()).strip()

def str_sim(a,b):
    a=norm(a);b=norm(b)
    if not a or not b:return 0
    if a==b:return 1
    if a in b or b in a:return .8
    dp=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        q=[i]
        for j,cb in enumerate(b,1):q.append(min(q[-1]+1,dp[j]+1,dp[j-1]+(ca!=cb)))
        dp=q
    return max(0,1-dp[-1]/max(len(a),len(b)))

def constrained(label,key,raw,prev=''):
    raw=(raw or '').strip()
    if raw=='"' and prev!='': return prev,.99,'fort','repeat'
    if not raw:return '',.99,'fort','empty'
    c=label
    if c=='Sexe':
        t=norm(raw)
        if re.search(r'(^|\s)f(\s|$)',t) or 'fem' in t or '♀' in raw:return 'F',.92,'fort','rule'
        if re.search(r'(^|\s)m(\s|$)',t) or 'male' in t or '♂' in raw:return 'M',.92,'fort','rule'
        if ('o' in t or '0' in t) and ('+' in t or 'x' in t):return 'F',.70,'moyen','symbol'
    if c=='Manip.':
        t=re.sub(r'[^A-Za-z]','',raw).upper()
        if re.fullmatch(r'[A-Z]{2,3}',t):return t,.90,'fort','rule'
        if t:return t,.30,'faible','rule'
    if c=='Heure Manip':
        m=re.search(r'\b([01]?\d|2[0-3])\s*[:hH.]\s*([0-5]?\d)\b',raw)
        if m:return f'{int(m.group(1)):02d}:{int(m.group(2)):02d}',.92,'fort','regex'
    if c in NUM_RULES:
        pat=NUM_PATTERNS.get(c,NUM_PATTERN_DEFAULT)
        m=re.search(pat,raw.replace('O','0').replace('o','0'))
        if m:
            token=m.group().replace('.',',')
            v=float(token.replace(',','.')); lo,hi=NUM_RULES[c]
            if lo<=v<=hi:return token,.92,'fort','range'
            return token,.30,'faible','range'
        return '',.10,'faible','numeric_missing'
    ekey=resolve_key(key)   # 'taxon' -> Taxon_Metropole / Taxon_Antilles selon la région choisie
    if ekey in CLOSED_SET_KEYS:
        vals=CLOSED_SET_KEYS[ekey]; nr=norm(raw)
        if nr in [norm(v) for v in vals]:
            return vals[[norm(v) for v in vals].index(nr)],.95,'fort','enum'
        hits=[v for v in vals if norm(v) in nr or nr in norm(v)]
        if len(hits)==1:return hits[0],.72,'moyen','enum'
        return '',.15,'faible','enum_missing'
    return raw,.45,'faible','raw'

class App:
    def __init__(self,root):
        self.root=root; root.title('Chiro OCR — gestion des fiches');root.geometry('1450x900')
        ensure_refs(); self.ocr=None; self.session=None; self.rows=[]; self.images=[]; self.manual_quads={}; self._tesseract_warned=False; self._last_ocr_conf=None; self._last_ocr_source=None
        self.build()
        self.refresh_history(); self.refresh_refs()

    def build(self):
        top=ttk.Frame(self.root,padding=8);top.pack(fill='x')
        ttk.Button(top,text='📷 Charger des JPG',command=self.load_images).pack(side='left',padx=3)
        ttk.Label(top,text='Modèle de fiche :').pack(side='left',padx=(12,2))
        self.template_var=tk.StringVar(value=cfg.template_name)
        template_box=ttk.Combobox(top,textvariable=self.template_var,values=sorted(TEMPLATES.keys()),state='readonly',width=16)
        template_box.pack(side='left',padx=2)
        template_box.bind('<<ComboboxSelected>>',self.on_template_change)
        # La colonne reste « Taxon » à l'écran ; la région choisit le référentiel interne
        # (Taxon_Metropole ou Taxon_Antilles) pour le décodage, la validation et l'entraînement.
        ttk.Label(top,text='Taxons :').pack(side='left',padx=(12,2))
        self.region_var=tk.StringVar(value=region.name)
        region_box=ttk.Combobox(top,textvariable=self.region_var,values=REGIONS,state='readonly',width=11)
        region_box.pack(side='left',padx=2)
        region_box.bind('<<ComboboxSelected>>',self.on_region_change)
        ttk.Button(top,text='▶ Décoder',command=self.decode_all).pack(side='left',padx=(12,3))
        ttk.Label(top,text='Lignes à décoder (0 = auto) :').pack(side='left',padx=(12,2))
        self.nlines=tk.IntVar(value=0)
        ttk.Spinbox(top,from_=0,to=25,textvariable=self.nlines,width=4).pack(side='left',padx=2)
        ttk.Button(top,text='💾 Enregistrer CSV',command=self.save_csv).pack(side='left',padx=3)
        ttk.Button(top,text='↻ Actualiser',command=self.refresh_all).pack(side='left',padx=3)
        self.template_info=ttk.Label(top,font=('Segoe UI',11,'bold'));self.template_info.pack(side='left',padx=(12,0))
        self.update_template_info()
        self.nb=ttk.Notebook(self.root);self.nb.pack(fill='both',expand=True)
        self.tab_data=ttk.Frame(self.nb);self.nb.add(self.tab_data,text='Données')
        self.tab_hist=ttk.Frame(self.nb);self.nb.add(self.tab_hist,text='Historique sessions')
        self.tab_refs=ttk.Frame(self.nb);self.nb.add(self.tab_refs,text='Référentiels')
        self.tab_log=ttk.Frame(self.nb);self.nb.add(self.tab_log,text='Journal')
        self.build_data();self.build_history();self.build_refs();self.build_log()

    def update_template_info(self):
        self.template_info.config(text=f'   {cfg.template_name} : {cfg.N_ROWS} lignes × {len(cfg.COLUMNS)} colonnes')

    def on_template_change(self,event=None):
        name=self.template_var.get()
        if name==cfg.template_name:return
        if self.rows:
            if not messagebox.askyesno('Changer de modèle de fiche',
                    f"Changer pour '{name}' effacera les données actuellement décodées (non enregistrées). Continuer ?"):
                self.template_var.set(cfg.template_name);return
        cfg.set_template(name)
        self.rows=[]; self.manual_quads={}
        self.update_template_info()
        self.rebuild_header()
        self.populate()
        self.log(f'Modèle de fiche changé : {name} ({cfg.N_ROWS} lignes × {len(cfg.COLUMNS)} colonnes).')

    def on_region_change(self,event=None):
        name=self.region_var.get()
        if name==region.name:return
        if self.rows:
            if not messagebox.askyesno('Changer de référentiel de taxons',
                    f"Changer pour '{name}' effacera les données actuellement décodées (non enregistrées), "
                    "car les taxons doivent être décodés avec le bon référentiel. Continuer ?"):
                self.region_var.set(region.name);return
        region.set_region(name)
        self.rows=[]
        self.populate()
        self.log(f'Référentiel de taxons changé : {name} ({region.entity("taxon")}).')

    def rebuild_header(self):
        for w in getattr(self,'header_widgets',[]):w.destroy()
        self.header_widgets=[]
        for ci,c in enumerate(cfg.COLUMNS):
            w=max(4,int(c.width_mm*0.9))
            lbl=tk.Label(self.grid_frame,text=c.label,width=w,anchor='center',relief='ridge',borderwidth=1,
                         background='#d9d9d9',font=('Segoe UI',9,'bold'))
            lbl.grid(row=0,column=ci,sticky='nsew')
            self.header_widgets.append(lbl)

    def build_data(self):
        outer=ttk.Frame(self.tab_data);outer.pack(fill='both',expand=True)
        self.grid_canvas=tk.Canvas(outer,background='white',highlightthickness=0)
        vsb=ttk.Scrollbar(outer,orient='vertical',command=self.grid_canvas.yview)
        hsb=ttk.Scrollbar(outer,orient='horizontal',command=self.grid_canvas.xview)
        self.grid_canvas.configure(yscrollcommand=vsb.set,xscrollcommand=hsb.set)
        self.grid_canvas.grid(row=0,column=0,sticky='nsew')
        vsb.grid(row=0,column=1,sticky='ns');hsb.grid(row=1,column=0,sticky='ew')
        outer.grid_rowconfigure(0,weight=1);outer.grid_columnconfigure(0,weight=1)
        self.grid_frame=tk.Frame(self.grid_canvas,background='white')
        self.grid_canvas.create_window((0,0),window=self.grid_frame,anchor='nw')
        self.grid_frame.bind('<Configure>',lambda e:self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox('all')))
        self.grid_canvas.bind('<MouseWheel>',lambda e:self.grid_canvas.yview_scroll(int(-e.delta/120),'units'))
        self.header_widgets=[]
        self.rebuild_header()
        self.cell_widgets={}   # (row_idx,key) -> Label affiché
        self.validated=set();self.corrected=set()   # {(row_idx,key)} déjà validées (vert clair) ou corrigées (vert foncé)
        self.training_records={}   # (row_idx,key) -> chemin relatif (dans training_data/) de l'exemple enregistré
        self.status=ttk.Label(self.tab_data,text='Aucune donnée');self.status.pack(side='bottom',fill='x')
        ttk.Label(self.tab_data,text="Clic = valider (vert clair) / re-cliquer pour retirer · Double-clic = corriger (vert foncé)",
                  foreground='#555555').pack(side='bottom',fill='x')

    def build_history(self):
        self.hist=ttk.Treeview(self.tab_hist,columns=('session','date','photos','rows'),show='headings')
        for k,t in zip(('session','date','photos','rows'),('Session','Date','Photos','Lignes')):self.hist.heading(k,text=t);self.hist.column(k,width=250)
        self.hist.pack(fill='both',expand=True)

    def build_refs(self):
        bar=ttk.Frame(self.tab_refs);bar.pack(fill='x',padx=8,pady=8)
        ttk.Button(bar,text='Gérer Manip.csv',command=lambda:self.edit_ref('Manip.csv',['Initiales','Nom'])).pack(side='left',padx=4)
        ttk.Button(bar,text='Gérer taxons.csv',command=lambda:self.edit_ref('taxons.csv',['Code','Espece'])).pack(side='left',padx=4)
        self.ref_info=ttk.Label(self.tab_refs,text='');self.ref_info.pack(fill='x',padx=8)

    def build_log(self):
        self.logbox=tk.Text(self.tab_log,wrap='word');self.logbox.pack(fill='both',expand=True)

    def log(self,msg):
        self.logbox.insert('end',datetime.now().strftime('%H:%M:%S ') + msg+'\n');self.logbox.see('end')

    def load_images(self):
        files=filedialog.askopenfilenames(title='Choisir les fiches JPG',filetypes=[('Images JPG','*.jpg *.jpeg'),('Toutes les images','*.jpg *.jpeg *.png *.tif')])
        if not files:return
        self.session=SESSIONS_DIR/datetime.now().strftime('%Y%m%d_%H%M%S');self.session.mkdir(parents=True,exist_ok=True)
        self.images=list(map(Path,files));
        for p in self.images:shutil.copy2(p,self.session/p.name)
        (self.session/'session.json').write_text(json.dumps({'created':datetime.now().isoformat(),'photos':[p.name for p in self.images]},ensure_ascii=False,indent=2),encoding='utf8')
        self.log(f'{len(self.images)} photo(s) chargée(s) dans {self.session.name}')
        for p in self.images:
            self.review_quad(p)
        self.status.config(text=f'{len(self.images)} photo(s) prête(s) au décodage')
        self.refresh_history()

    def review_quad(self,path):
        # Affiche la photo avec le contour détecté automatiquement (rouge) ; l'utilisateur peut
        # faire glisser les 4 coins puis valider. Le contour accepté (repère de l'image ORIGINALE,
        # avant rectification) est mémorisé dans self.manual_quads et sera utilisé au décodage à
        # la place d'une nouvelle détection automatique pour cette photo.
        img_bgr=load_img(path)
        quad,ok=page_quad(img_bgr)
        H,W=img_bgr.shape[:2]
        try:
            pil_img=PILImage.open(path).convert('RGB')
            if pil_img.size!=(W,H):pil_img=pil_img.resize((W,H))
        except Exception as e:
            self.log(f'Aperçu impossible pour {path.name} ({e}) — contour détecté automatiquement conservé.')
            self.manual_quads[str(path)]=quad;return
        max_w,max_h=900,680
        scale=min(max_w/W,max_h/H,1.0)
        disp_w,disp_h=max(1,int(W*scale)),max(1,int(H*scale))
        disp_img=pil_img.resize((disp_w,disp_h),PILImage.LANCZOS)
        photo_img=ImageTk.PhotoImage(disp_img)

        win=tk.Toplevel(self.root);win.title(f'Vérifier le contour — {path.name}');win.transient(self.root);win.grab_set();win.resizable(False,False)
        msg='Contour détecté automatiquement (en rouge).' if ok else \
            "Contour NON détecté automatiquement (toute la photo est proposée par défaut) : ajustez les 4 coins."
        ttk.Label(win,text=f"{msg}\nFaites glisser un coin pour le corriger, puis validez.",justify='center',padding=(8,8,8,4)).pack()
        canvas=tk.Canvas(win,width=disp_w,height=disp_h,cursor='crosshair',highlightthickness=0)
        canvas.pack(padx=8,pady=4)
        canvas.create_image(0,0,anchor='nw',image=photo_img)
        canvas.image=photo_img  # garder une référence vivante, sinon Tk la libère et l'image disparaît

        pts=[[float(x)*scale,float(y)*scale] for x,y in quad]
        R=9
        def redraw():
            canvas.delete('overlay')
            flat=[coord for p in pts for coord in p]
            canvas.create_polygon(flat,outline='#ff3b30',fill='',width=2,tags='overlay')
            for x,y in pts:
                canvas.create_oval(x-R,y-R,x+R,y+R,outline='white',width=2,fill='#ff3b30',tags='overlay')
        redraw()

        drag={'i':None}
        def hit(event):
            for i,(x,y) in enumerate(pts):
                if (event.x-x)**2+(event.y-y)**2<=(R*1.8)**2:return i
            return None
        def on_press(event):drag['i']=hit(event)
        def on_drag(event):
            i=drag['i']
            if i is None:return
            pts[i][0]=min(max(0,event.x),disp_w);pts[i][1]=min(max(0,event.y),disp_h)
            redraw()
        def on_release(event):drag['i']=None
        canvas.bind('<ButtonPress-1>',on_press);canvas.bind('<B1-Motion>',on_drag);canvas.bind('<ButtonRelease-1>',on_release)

        result={}
        def on_validate():
            result['quad']=[[x/scale,y/scale] for x,y in pts];win.destroy()
        def on_reset():
            nonlocal pts
            q2,_=page_quad(img_bgr)
            pts=[[float(x)*scale,float(y)*scale] for x,y in q2]
            redraw()
        btns=ttk.Frame(win);btns.pack(pady=(4,10))
        ttk.Button(btns,text='↺ Redétecter',command=on_reset).pack(side='left',padx=6)
        ttk.Button(btns,text='✓ Valider ce contour',command=on_validate).pack(side='left',padx=6)
        win.protocol('WM_DELETE_WINDOW',on_validate)  # fermer la fenêtre = valider l'état actuel (comportement sûr)
        win.wait_window()
        self.manual_quads[str(path)]=np.array(result.get('quad',quad),dtype=np.float32)

    def get_ocr(self):
        if self.ocr is None:
            try:
                self.ocr=LocalHandwritingOCR(str(MODEL_DIR));self.ocr.load()
                self.log('Modèle OCR manuscrit local chargé.')
            except Exception as e:
                self.log('Modèle TrOCR local indisponible : '+str(e)); self.ocr=False
        return self.ocr

    def decode_all(self):
        if not self.images:
            messagebox.showwarning('Décodage','Chargez au moins une photo.');return
        self.rows=[]
        if not self.get_ocr():
            messagebox.showerror('TrOCR indisponible', "Le modèle OCR manuscrit local ne peut pas être chargé. Aucun décodage n'a été effectué.")
            return
        override=int(self.nlines.get())  # 0 = détection automatique des lignes utiles
        for photo in self.images:
            try:self.decode_photo(photo,None if override==0 else max(1,min(25,override)))
            except Exception as e:self.log(f'ERREUR {photo.name}: {e}');self.log(traceback.format_exc())
        self.populate()
        self.log(f'Décodage terminé : {len(self.rows)} lignes.')
        # Vérifie qu'au moins une partie des cellules a réellement été lue,
        # pour ne pas afficher un faux succès quand aucun moteur OCR ne fonctionne.
        data_keys=[c.key for c in cfg.COLUMNS if c.key!='n']
        total=sum(1 for r in self.rows for k in data_keys if k in r)
        filled=sum(1 for r in self.rows for k in data_keys if k in r and str(r.get(k,'')).strip())
        if total and filled==0:
            msg=('Aucun texte n\'a pu être lu : ni le modèle OCR local, ni le repli Tesseract ne fonctionnent.\n'
                 'Consultez le journal ci-dessus (ex. "sentencepiece" manquant, ou Tesseract non installé).')
            self.log('ATTENTION : '+msg)
            messagebox.showwarning('Décodage vide',msg)

    def decode_photo(self,path,nlines=None):
        img=load_img(path); can,quad,ok=rectify(img,quad=self.manual_quads.get(str(path)))
        out=self.session/'processed'/path.stem;out.mkdir(parents=True,exist_ok=True)
        # Diagnostics: the detected page quadrilateral, canonical image and exact 27x25 grid.
        qimg=img.copy(); qq=quad.astype(np.int32).reshape((-1,1,2)); cv2.polylines(qimg,[qq],True,(0,0,255),5); cv2.imwrite(str(out/'00_page_quad.jpg'),qimg)
        cv2.imwrite(str(out/'01_canonical.jpg'),can)
        xs,ys=fixed_boundaries(); draw_grid(can,xs,ys,out/'02_decoupage.jpg')
        if nlines is None:
            nlines=count_used_rows(can,xs,ys)
            self.log(f'{path.name} : {nlines} ligne(s) utile(s) détectée(s) automatiquement.')
        with open(out/'geometry.json','w',encoding='utf8') as f:json.dump({'quad':quad.tolist(),'perspective':ok,'canonical_width':cfg.CANONICAL_WIDTH,'canonical_height':cfg.CANONICAL_HEIGHT,'x_boundaries':xs,'y_boundaries':ys,'column_x_mm':getattr(cfg,'COLUMN_X_MM',None),
                'column_widths_mm':[c.width_mm for c in cfg.COLUMNS],
                'rows_total':cfg.N_ROWS,'rows_processed':nlines},f,indent=2)
        prev={}
        for r in range(nlines):
            row={'photo':path.name,'ligne':r+1}
            for ci,c in enumerate(cfg.COLUMNS):
                if c.key=='n': row[c.key]=str(r+1);continue
                x1,x2=column_x_bounds(xs,ci); y1,y2=row_y_bounds(ys,r)
                cell=crop_cell(can,x1,x2,y1,y2); cp=out/f'{r+1:02d}_{c.key}.png';cv2.imwrite(str(cp),cell)
                raw=self.ocr_read(c.key,cp)
                val,conf,level,source=constrained(c.label,c.key,raw,prev.get(c.key,''))
                if self._last_ocr_source=='closed_set':
                    # Le classifieur a déjà produit une classe valide (ou vide).
                    # On conserve sa probabilité comme confiance du moteur, puis on
                    # laisse la règle de domaine confirmer la valeur.
                    mconf=float(self._last_ocr_conf or 0.0)
                    if raw=='"' and prev.get(c.key,''):
                        val=prev[c.key]; source='repeat'; conf=.99; level='fort'
                    elif raw=='':
                        val=''; source='empty'; conf=max(.95,mconf); level='fort'
                    else:
                        conf=mconf; level='fort' if mconf>=.75 else ('moyen' if mconf>=.50 else 'faible'); source='closed_set'
                if c.key=='manip': val,conf,level=self.resolve_observateur(val)
                # Taxon : plus de rapprochement flou avec le petit référentiel local — le décodage
                # contraint garantit déjà un code officiel de config/knowledge_taxons_<Région>.json (cf. ocr_read).
                row[c.key]=val;row[c.key+'__raw']=raw;row[c.key+'__confidence']=conf;row[c.key+'__level']=level;row[c.key+'__image']=str(cp)
                if val:prev[c.key]=val
            self.rows.append(row)

    def ocr_read(self,key,path):
        # First decide whether the cell contains actual ink. TrOCR is generative and
        # may hallucinate text in a blank cell, so blank detection happens BEFORE OCR.
        # Loaded in colour (not grayscale): some columns are printed on a tinted (blue/pink)
        # background, and converting to grayscale first hid the ink/background contrast there.
        img=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if img is None:return ''
        h,w=img.shape[:2]; m=max(3,int(min(h,w)*.08)); g=img[m:h-m,m:w-m]
        g=cv2.GaussianBlur(g,(3,3),0)
        ink,largest=ink_ratio_and_largest(g)
        min_area=max(15,int(g.shape[0]*g.shape[1]*BLANK_MIN_AREA_RATIO))
        # Very low ink, or only a speck too small to be a real stroke, is a blank cell;
        # do not ask a generative OCR model to guess.
        if ink < BLANK_INK_RATIO or largest < min_area:
            return ''
        o=self.get_ocr()
        if not o: raise RuntimeError('Aucun moteur OCR disponible')
        try:
            # Priorité absolue aux classifieurs entraînés localement pour les colonnes
            # à choix fermé. Ils savent notamment reconnaître explicitement la classe vide.
            ekey=resolve_key(key)   # Taxon -> classifieur models/closed_set/Taxon_<Région>.pt
            if ekey in getattr(o,'closed_models',{}):
                value,conf=o.closed_set_predict(path,ekey)
                self._last_ocr_conf=conf; self._last_ocr_source='closed_set'
                return value
            self._last_ocr_conf=None; self._last_ocr_source='trocr'
            if key=='manip':
                # Sans manip.pt : initiales manuscrites, puis résolution dans observateurs.json.
                return o.raw(path,charset='ABCDEFGHIJKLMNOPQRSTUVWXYZ',min_len=2,max_len=3)[0]
            if key=='heure':
                # Heure structurée : 4 chiffres. Si des modèles digitaux sont présents, ils sont
                # essayés en priorité ; sinon TrOCR contraint reste le secours.
                if hasattr(o,'predict_hour'):
                    value,conf,source=o.predict_hour(path)
                    if source=='closed_set_hour':
                        self._last_ocr_conf=conf; self._last_ocr_source='closed_set_hour'
                        return value
                return o.raw(path,charset='0123456789:hH.',min_len=3,max_len=5)[0]
            return o.raw(path,numeric=key in NUMERIC_OCR_KEYS,closed_set=CLOSED_SET_KEYS.get(ekey))[0]
        except Exception as e:
            self.log(f'OCR {key}: {e}')
            return ''

    def load_refs(self):
        return {'manip':read_csv(REF_DIR/'Manip.csv'),'taxons':read_csv(REF_DIR/'taxons.csv')}

    def match_ref(self,raw,rows,fields):
        if not raw:return '',.99,'fort'
        def sim(a,b):
            a=norm(a);b=norm(b)
            if not a or not b:return 0
            if a==b:return 1
            if a in b or b in a:return .8
            # edit similarity
            dp=list(range(len(b)+1))
            for i,ca in enumerate(a,1):
                q=[i]
                for j,cb in enumerate(b,1):q.append(min(q[-1]+1,dp[j]+1,dp[j-1]+(ca!=cb)))
                dp=q
            return max(0,1-dp[-1]/max(len(a),len(b)))
        best=(-1,None)
        for row in rows:
            for f in fields:
                s=sim(raw,row.get(f,''))
                if s>best[0]:best=(s,row)
        if best[1] is None or best[0]<.55:return raw,.30,'faible'
        return (best[1].get(fields[0],raw),min(1,best[0]),'fort' if best[0]>=.85 else 'moyen')

    def resolve_observateur(self,initials):
        # 'initials' vient déjà de constrained() : 2-3 lettres majuscules si le format est respecté.
        # observateurs.json fait la correspondance initiales -> nom complet ; plusieurs initiales
        # peuvent pointer vers la même personne (ex. 'MT' et 'TM'), la case garde toujours le nom complet.
        if not initials:return '',.99,'fort'
        key=initials.strip().upper()
        name=OBSERVATEURS.get(key)
        if name:return name,.95,'fort'
        if not OBSERVATEURS:return initials,.30,'faible'
        s,name=max(((str_sim(key,k),v) for k,v in OBSERVATEURS.items()),key=lambda z:z[0])
        if s>=.5:return name,min(.85,.5+s/2),('fort' if s>=.8 else 'moyen')
        return initials,.25,'faible'

    CELL_COLORS={'faible':'#ffd9d9','moyen':'#fff6c8','fort':'#ffffff','':'#ffffff'}
    COLOR_VALIDATED='#bdf7bd'   # vert clair : cliqué pour valider tel quel
    COLOR_CORRECTED='#2e7d32'   # vert foncé : cliqué puis modifié manuellement

    def cell_color(self,row_idx,key,row):
        if (row_idx,key) in self.corrected:return self.COLOR_CORRECTED
        if (row_idx,key) in self.validated:return self.COLOR_VALIDATED
        return self.CELL_COLORS.get(row.get(key+'__level',''),'#ffffff')

    def populate(self):
        for w in self.cell_widgets.values():w.destroy()
        self.cell_widgets={};self.validated=set();self.corrected=set();self.training_records={}
        for ri,row in enumerate(self.rows):
            for ci,c in enumerate(cfg.COLUMNS):
                w=max(4,int(c.width_mm*0.9))
                lbl=tk.Label(self.grid_frame,text=row.get(c.key,''),width=w,anchor='center',relief='solid',borderwidth=1,
                             background=self.cell_color(ri,c.key,row),cursor='hand2' if c.key!='n' else 'arrow')
                lbl.grid(row=ri+1,column=ci,sticky='nsew')
                if c.key!='n':
                    lbl.bind('<Button-1>',lambda e,ri=ri,c=c:self.validate_cell(ri,c))
                    lbl.bind('<Double-Button-1>',lambda e,ri=ri,c=c:self.edit_cell_grid(ri,c))
                self.cell_widgets[(ri,c.key)]=lbl
        self.grid_frame.update_idletasks();self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox('all'))
        self.refresh_status()

    def refresh_status(self):
        keys=[c.key for c in cfg.COLUMNS if c.key!='n']
        total=len(self.rows)*len(keys)
        done=len(self.validated)+len(self.corrected)
        weak=sum(1 for ri,r in enumerate(self.rows) for k in keys if r.get(k+'__level')=='faible' and (ri,k) not in self.validated and (ri,k) not in self.corrected)
        self.status.config(text=f'{len(self.rows)} ligne(s) — {done}/{total} case(s) traitée(s) ({len(self.validated)} validée(s), {len(self.corrected)} corrigée(s)) · {weak} à confiance faible restante(s).')

    def validate_cell(self,ri,c):
        key=c.key; row=self.rows[ri]
        if (ri,key) in self.validated or (ri,key) in self.corrected:
            # Re-clic sur une case déjà colorée : on retire la validation/correction — la case
            # redevient sans couleur particulière et l'exemple est retiré du jeu d'entraînement.
            self.validated.discard((ri,key));self.corrected.discard((ri,key))
            rel=self.training_records.pop((ri,key),None)
            if rel:self.remove_training_example(rel)
            w=self.cell_widgets.get((ri,key))
            if w:w.configure(background=self.cell_color(ri,key,row),foreground='black')
            self.refresh_status();return
        val=row.get(key,''); img_path=row.get(key+'__image','')
        self.validated.add((ri,key))
        w=self.cell_widgets.get((ri,key))
        if w:w.configure(background=self.COLOR_VALIDATED,foreground='black')
        if img_path:
            rel=self.record_training_example(key,val,img_path,'validated')
            if rel:self.training_records[(ri,key)]=rel
        self.refresh_status()

    def ask_value(self,label,current,options=None,option_labels=None):
        # Sans liste fermée : saisie libre au clavier (comme avant).
        if options is None:
            return simpledialog.askstring('Correction',f'{label} — valeur :',initialvalue=current,parent=self.root)
        # Avec liste fermée : menu déroulant uniquement, pour ne jamais pouvoir saisir une valeur invalide.
        # Affiche "code — description" quand disponible (knowledge_donnees.json / knowledge_taxons.json),
        # mais seul le code est conservé comme valeur de la case.
        option_labels=option_labels or {}
        display=[option_labels.get(o,o) if o else '(vide)' for o in options]
        inv=dict(zip(display,options))
        win=tk.Toplevel(self.root);win.title('Correction');win.transient(self.root);win.resizable(False,False);win.grab_set()
        ttk.Label(win,text=f'{label} — valeur :',padding=(12,12,12,4)).pack(anchor='w')
        cur_display=option_labels.get(current,current) if current in options else display[0]
        var=tk.StringVar(value=cur_display)
        width=min(60,max(10,max(len(d) for d in display)+2))
        combo=ttk.Combobox(win,textvariable=var,values=display,state='readonly',width=width)
        combo.pack(padx=12,pady=4,fill='x');combo.focus_set()
        result={'value':None}
        def on_ok(event=None):result['value']=inv.get(var.get(),var.get());win.destroy()
        def on_cancel():win.destroy()
        btns=ttk.Frame(win);btns.pack(pady=(4,12))
        ttk.Button(btns,text='OK',command=on_ok).pack(side='left',padx=4)
        ttk.Button(btns,text='Annuler',command=on_cancel).pack(side='left',padx=4)
        win.bind('<Return>',on_ok);win.protocol('WM_DELETE_WINDOW',on_cancel)
        win.update_idletasks()
        x=self.root.winfo_rootx()+(self.root.winfo_width()-win.winfo_width())//2
        y=self.root.winfo_rooty()+(self.root.winfo_height()-win.winfo_height())//2
        win.geometry(f'+{max(0,x)}+{max(0,y)}')
        win.wait_window()
        return result['value']

    def edit_cell_grid(self,ri,c):
        key=c.key; row=self.rows[ri]; old=row.get(key,'')
        ekey=resolve_key(key)
        options=CLOSED_SET_KEYS.get(ekey)
        if key=='manip':
            # Menu déroulant des noms complets (observateurs.json) : la case garde toujours
            # le nom complet, jamais les initiales, quelle que soit l'orthographe manuscrite.
            new=self.ask_value(c.label,old,options=['']+OBSERVATEUR_NAMES)
        elif options is not None:
            new=self.ask_value(c.label,old,options=['']+list(options),option_labels=CLOSED_SET_LABELS.get(ekey))
        else:
            new=self.ask_value(c.label,old,options=None)
            if new is not None and key in DECIMAL_KEYS:
                new=new.replace('.',',')  # cohérence avec le format attendu (virgule décimale)
        if new is None:return
        row[key]=new;row[key+'__confidence']=1.;row[key+'__level']='fort'
        old_rel=self.training_records.pop((ri,key),None)
        if old_rel:self.remove_training_example(old_rel)  # ne garder que la dernière valeur validée pour cette case
        self.validated.discard((ri,key));self.corrected.add((ri,key))
        w=self.cell_widgets.get((ri,key))
        if w:w.configure(text=new,background=self.COLOR_CORRECTED,foreground='white')
        img_path=row.get(key+'__image','')
        if img_path:
            rel=self.record_training_example(key,new,img_path,'corrected')
            if rel:self.training_records[(ri,key)]=rel
        self.refresh_status()

    def record_training_example(self,key,value,image_path,source):
        # Copie la case telle que vue par l'OCR (même image, même recadrage) dans le corpus
        # d'entraînement « local » (training_data/local), avec sa valeur validée par un humain.
        # La clé enregistrée est l'entité interne (Taxon -> Taxon_Metropole / Taxon_Antilles selon la région).
        # Renvoie le chemin relatif (dans training_data/local/) de l'exemple enregistré, pour pouvoir
        # le retrouver et le retirer plus tard si la validation est annulée.
        ekey=resolve_key(key)
        try:return training_store.record_example(ekey,value,image_path,source)
        except Exception as e:self.log(f"Entraînement {ekey} : enregistrement impossible ({e})");return None

    def remove_training_example(self,rel_path):
        # Retire un exemple précédemment enregistré dans « local » (l'image ET sa ligne dans le manifeste) —
        # utilisé quand une validation/correction est annulée d'un clic : l'exemple ne doit plus
        # servir à l'entraînement. Le corpus « global » n'est jamais modifié.
        try:training_store.remove_example(rel_path)
        except Exception as e:self.log(f"Entraînement : suppression impossible ({e})")

    def save_csv(self):
        if not self.rows:
            messagebox.showwarning('CSV','Aucune donnée à enregistrer.');return
        p=self.session/'donnees_decodees.csv' if self.session else filedialog.asksaveasfilename(defaultextension='.csv',filetypes=[('CSV','*.csv')])
        if not p:return
        fields=[c.key for c in cfg.COLUMNS]
        write_csv(Path(p),[{k:r.get(k,'') for k in fields} for r in self.rows],fields)
        self.log(f'CSV enregistré : {p}')
        messagebox.showinfo('CSV',f'CSV enregistré dans\n{p}')

    def refresh_history(self):
        if not hasattr(self,'hist'):return
        self.hist.delete(*self.hist.get_children())
        for p in sorted(SESSIONS_DIR.iterdir(),reverse=True):
            if p.is_dir() and (p/'session.json').exists():
                d=json.loads((p/'session.json').read_text(encoding='utf8'));self.hist.insert('', 'end',values=(p.name,d.get('created',''),len(d.get('photos',[])),sum(1 for _ in p.rglob('*_confidence') )))

    def refresh_refs(self):
        if hasattr(self,'ref_info'):
            self.ref_info.config(text=f'Manip.csv : {len(read_csv(REF_DIR/"Manip.csv"))} entrées    |    taxons.csv : {len(read_csv(REF_DIR/"taxons.csv"))} entrées')

    def edit_ref(self,name,fields):
        p=REF_DIR/name;rows=read_csv(p)
        win=tk.Toplevel(self.root);win.title(name);win.geometry('800x500')
        tree=ttk.Treeview(win,columns=fields,show='headings');tree.pack(fill='both',expand=True)
        for f in fields:tree.heading(f,text=f);tree.column(f,width=350)
        for r in rows:tree.insert('', 'end',values=[r.get(f,'') for f in fields])
        def add():
            vals=[simpledialog.askstring(f,'Nouvelle entrée',f'{f} :',parent=win) or '' for f in fields]
            if any(vals):tree.insert('', 'end',values=vals)
        def save():
            rr=[dict(zip(fields,tree.item(i,'values'))) for i in tree.get_children()];write_csv(p,rr,fields);self.refresh_refs();self.log(f'{name} mis à jour ({len(rr)} entrées)');win.destroy()
        b=ttk.Frame(win);b.pack(fill='x');ttk.Button(b,text='Ajouter',command=add).pack(side='left',padx=5,pady=5);ttk.Button(b,text='Enregistrer',command=save).pack(side='right',padx=5,pady=5)

    def refresh_all(self):self.refresh_history();self.refresh_refs();self.log('Actualisation terminée.')

if __name__=='__main__':
    root=tk.Tk();App(root);root.mainloop()
