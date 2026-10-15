from pathlib import Path
import json,re,unicodedata,cv2,numpy as np
import settings

def norm(s):
 s='' if s is None else str(s); s=unicodedata.normalize('NFD',s); s=''.join(c for c in s if unicodedata.category(c)!='Mn'); return re.sub(r'[^a-z0-9+/?.,:-]+',' ',s.lower()).strip()
def lev(a,b):
 a,b=norm(a),norm(b); p=list(range(len(b)+1))
 for i,ca in enumerate(a,1):
  q=[i]
  for j,cb in enumerate(b,1): q.append(min(q[-1]+1,p[j]+1,p[j-1]+(ca!=cb)))
  p=q
 return p[-1]
def sim(a,b):
 a,b=norm(a),norm(b); return 0.0 if not a or not b else max(0.0,1-lev(a,b)/max(len(a),len(b)))
class LocalHandwritingOCR:
 def __init__(self,model_dir='models/trocr-handwritten',data_file=None):
  # Les taxons dépendent de la région active (settings.region) : voir la propriété `taxons`.
  self.model_dir=Path(model_dir); self.data=json.loads(Path(data_file or settings.CONFIG_DIR/'knowledge_donnees.json').read_text(encoding='utf-8')); self.processor=self.model=self.device=None; self.closed_models={}; self.closed_classes={}; self.closed_img_size={}; self._trocr_error=None
 @property
 def taxons(self):
  # Référentiel de taxons de la région active (Taxon_Metropole, Taxon_Antilles...).
  return settings.region.taxons
 def load(self):
  # Les classifieurs fermés (*.pt) sont prioritaires et indépendants de TrOCR.
  # On les charge en premier : une panne du tokenizer TrOCR ne doit plus empêcher
  # l''utilisation des colonnes pour lesquelles un modèle dédié existe.
  import torch
  import torch.nn as nn
  class TinyCNN(nn.Module):
   def __init__(self,n_classes):
    super().__init__()
    self.features=nn.Sequential(
     nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
     nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
     nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
    self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(64*8*8,128),nn.ReLU(),nn.Dropout(0.3),nn.Linear(128,n_classes))
   def forward(self,x): return self.classifier(self.features(x))

  closed_dir=self.model_dir.parent/'closed_set'
  if closed_dir.exists():
   for pt in sorted(closed_dir.glob('*.pt')):
    try:
     ckpt=torch.load(pt,map_location='cpu',weights_only=False)
     classes=list(ckpt['classes']); model=TinyCNN(len(classes)); model.load_state_dict(ckpt['state_dict']); model.eval()
     key=pt.stem; self.closed_models[key]=model; self.closed_classes[key]=classes; self.closed_img_size[key]=int(ckpt.get('img_size',64))
    except Exception as e:
     self._trocr_error=self._trocr_error or f'Classifieur {pt.name}: {e}'

  # Classifieurs optionnels spécialisés pour les quatre chiffres de l'heure.
  # Noms : heure_h10, heure_h1, heure_m10, heure_m1.
  self.hour_models={}; self.hour_classes={}; self.hour_img_size={}
  for hk in ('heure_h10','heure_h1','heure_m10','heure_m1'):
   pt=closed_dir/f'{hk}.pt'
   if pt.exists():
    try:
     ckpt=torch.load(pt,map_location='cpu',weights_only=False)
     classes=list(ckpt['classes']); model=TinyCNN(len(classes))
     model.load_state_dict(ckpt['state_dict']); model.eval()
     self.hour_models[hk]=model; self.hour_classes[hk]=classes; self.hour_img_size[hk]=int(ckpt.get('img_size',64))
    except Exception as e:
     self._trocr_error=self._trocr_error or f'Classifieur {pt.name}: {e}'

  # TrOCR reste disponible pour les colonnes sans classifieur dédié.
  try:
   from transformers import TrOCRProcessor,VisionEncoderDecoderModel
   if not (self.model_dir/'config.json').exists():
    raise FileNotFoundError(f'Modèle local absent: {self.model_dir}')
   self.processor=TrOCRProcessor.from_pretrained(self.model_dir,local_files_only=True,use_fast=False)
   self.model=VisionEncoderDecoderModel.from_pretrained(self.model_dir,local_files_only=True)
   self.device='cuda' if torch.cuda.is_available() else 'cpu'; self.model.to(self.device); self.model.eval()
  except Exception as e:
   self._trocr_error=e

  if not self.closed_models and self.model is None:
   raise RuntimeError(f'Aucun moteur OCR utilisable. TrOCR: {self._trocr_error}')

 def closed_set_predict(self,path,key):
  import torch
  if key not in self.closed_models:
   raise KeyError(f'Aucun modèle closed_set pour la colonne {key}')
  img=cv2.imread(str(path),cv2.IMREAD_COLOR)
  if img is None: return '',1.0
  n=self.closed_img_size[key]
  img=cv2.resize(img,(n,n),interpolation=cv2.INTER_AREA)
  img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
  x=torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
  model=self.closed_models[key]
  with torch.no_grad():
   prob=torch.softmax(model(x),dim=1)[0]
   idx=int(torch.argmax(prob).item()); conf=float(prob[idx].item())
  return self.closed_classes[key][idx],conf

 def predict_hour(self,path):
  # Découpe la cellule heure en quatre zones de chiffres. Le séparateur n'est pas classifié.
  # Le modèle est utilisé seulement si les quatre modèles sont disponibles.
  needed=('heure_h10','heure_h1','heure_m10','heure_m1')
  if not all(k in self.hour_models for k in needed):
   return '',0.0,'trocr'
  import torch
  img=cv2.imread(str(path),cv2.IMREAD_COLOR)
  if img is None:return '',0.0,'trocr'
  h,w=img.shape[:2]
  # Les marges internes sont déjà retirées par crop_cell. On réserve environ 17% à 42%,
  # 42% à 58% étant le séparateur ; les proportions restent ajustables.
  boxes=[
   (0,int(w*.25)), (int(w*.25),int(w*.43)),
   (int(w*.57),int(w*.75)), (int(w*.75),w)
  ]
  vals=[]; confs=[]
  for hk,(x1,x2) in zip(needed,boxes):
   patch=img[:,x1:x2]
   if patch.size==0:return '',0.0,'trocr'
   n=self.hour_img_size[hk]
   patch=cv2.resize(patch,(n,n),interpolation=cv2.INTER_AREA)
   patch=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
   x=torch.from_numpy(patch).permute(2,0,1).unsqueeze(0)
   with torch.no_grad():
    p=torch.softmax(self.hour_models[hk](x),dim=1)[0]
   idx=int(torch.argmax(p).item()); vals.append(self.hour_classes[hk][idx]); confs.append(float(p[idx].item()))
  raw=''.join(vals[:2])+':'+''.join(vals[2:])
  try:
   hh=int(raw[:2]); mm=int(raw[3:])
   if not (0<=hh<=23 and 0<=mm<=59): return '',float(np.mean(confs)),'trocr'
  except Exception:return '',float(np.mean(confs)),'trocr'
  return raw,float(np.mean(confs)),'closed_set_hour'

 def variants(self,path):
  img=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
  if img is None:return []
  img=cv2.copyMakeBorder(img,15,15,15,15,cv2.BORDER_CONSTANT,value=255); cl=cv2.createCLAHE(2.0,(8,8)).apply(img); bw=cv2.adaptiveThreshold(cl,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,11)
  return [cv2.cvtColor(x,cv2.COLOR_GRAY2RGB) for x in (img,cl,bw)]
 def _numeric_allowed_ids(self):
  if getattr(self,'_numeric_ids',None) is not None:return self._numeric_ids
  tok=self.processor.tokenizer; allowed_chars=set('0123456789,.- '); ids=set()
  for i in range(tok.vocab_size):
   try:s=tok.decode([i],skip_special_tokens=False)
   except Exception:continue
   if s=='' or all(ch in allowed_chars for ch in s):ids.add(i)
  for sp in (tok.bos_token_id,tok.eos_token_id,tok.pad_token_id,getattr(tok,'cls_token_id',None),getattr(tok,'sep_token_id',None)):
   if sp is not None:ids.add(sp)
  dsi=getattr(self.model.config,'decoder_start_token_id',None)
  if dsi is not None:ids.add(dsi)
  self._numeric_ids=sorted(ids);return self._numeric_ids
 def _charset_allowed_ids(self,charset):
  # Generic version of _numeric_allowed_ids for an arbitrary set of characters (e.g. uppercase
  # letters for initials, or digits+separators for a time). A space is always allowed too, since
  # byte-level BPE tokens often carry a leading space (" A" rather than "A").
  cache=getattr(self,'_charset_cache',None) or {}
  if charset in cache:return cache[charset]
  tok=self.processor.tokenizer; allowed=set(charset)|{' '}; ids=set()
  for i in range(tok.vocab_size):
   try:s=tok.decode([i],skip_special_tokens=False)
   except Exception:continue
   if s=='' or all(ch in allowed for ch in s):ids.add(i)
  for sp in (tok.bos_token_id,tok.eos_token_id,tok.pad_token_id,getattr(tok,'cls_token_id',None),getattr(tok,'sep_token_id',None)):
   if sp is not None:ids.add(sp)
  dsi=getattr(self.model.config,'decoder_start_token_id',None)
  if dsi is not None:ids.add(dsi)
  ids=sorted(ids); cache[charset]=ids; self._charset_cache=cache; return ids
 def _length_bounded_prefix_fn(self,allowed_ids,min_len=None,max_len=None):
  # Restricts generation to allowed_ids (a character class), and additionally forbids stopping
  # (eos) before min_len characters have been produced, and forces stopping at max_len.
  dsi=getattr(self.model.config,'decoder_start_token_id',None); eos=self.processor.tokenizer.eos_token_id
  allowed_no_eos=[i for i in allowed_ids if i!=eos] or allowed_ids
  tok=self.processor.tokenizer
  def fn(batch_id,input_ids):
   gen=input_ids.tolist()
   if gen and gen[0]==dsi:gen=gen[1:]
   if eos in gen:return [eos]
   text=tok.decode(gen,skip_special_tokens=True).strip()
   if max_len is not None and len(text)>=max_len:return [eos]
   if min_len is not None and len(text)<min_len:return allowed_no_eos
   return allowed_ids
  return fn
 def _closed_set_trie(self,values):
  # Builds a trie of token-id sequences for every allowed answer (plus the empty string,
  # so the model can still say "blank" instead of forcing a wrong guess). Both the plain
  # and the leading-space tokenisation are included since byte-level BPE tokenizers often
  # encode a word differently depending on whether it starts a sequence or not.
  cache=getattr(self,'_closed_set_cache',None) or {}
  key=tuple(values)
  if key in cache:return cache[key]
  tok=self.processor.tokenizer; seqs=set()
  for v in values:
   forms={v,' '+v} if v else {''}
   for s in forms:
    ids=tuple(tok(s,add_special_tokens=False).input_ids) if s else tuple()
    seqs.add(ids)
  trie={}; maxlen=0
  for seq in seqs:
   maxlen=max(maxlen,len(seq)); node=trie
   for tid in seq:node=node.setdefault(tid,{})
   node['__end__']=True
  cache[key]=(trie,maxlen); self._closed_set_cache=cache; return trie,maxlen
 def _closed_set_prefix_fn(self,trie):
  dsi=getattr(self.model.config,'decoder_start_token_id',None); eos=self.processor.tokenizer.eos_token_id
  def fn(batch_id,input_ids):
   gen=input_ids.tolist()
   if gen and gen[0]==dsi:gen=gen[1:]
   node=trie
   for tid in gen:
    if tid==eos:return [eos]
    node=node.get(tid)
    if node is None:return [eos]  # shouldn't happen; fail safe rather than crash
   allowed=[k for k in node if k!='__end__']
   if node.get('__end__'):allowed=allowed+[eos]
   return allowed or [eos]
  return fn
 def raw(self,path,numeric=False,closed_set=None,charset=None,min_len=None,max_len=None):
  import torch
  from PIL import Image
  if self.model is None:self.load()
  best=('',0.0)
  gen_kwargs=dict(max_new_tokens=12 if (numeric or charset) else 32,num_beams=4,return_dict_in_generate=True,output_scores=True)
  if closed_set is not None:
   values=sorted(set(closed_set)|{''})  # la case vide reste toujours une réponse valide
   trie,maxlen=self._closed_set_trie(values)
   gen_kwargs['prefix_allowed_tokens_fn']=self._closed_set_prefix_fn(trie)
   gen_kwargs['max_new_tokens']=maxlen+2
  elif charset is not None:
   allowed=self._charset_allowed_ids(charset)
   gen_kwargs['prefix_allowed_tokens_fn']=self._length_bounded_prefix_fn(allowed,min_len=min_len,max_len=max_len)
   if max_len is not None:gen_kwargs['max_new_tokens']=max_len+4
  elif numeric:
   allowed=self._numeric_allowed_ids()
   gen_kwargs['prefix_allowed_tokens_fn']=lambda batch_id,input_ids:allowed
  for arr in self.variants(path):
   px=self.processor(images=Image.fromarray(arr),return_tensors='pt').pixel_values.to(self.device)
   with torch.no_grad(): out=self.model.generate(px,**gen_kwargs)
   txt=self.processor.batch_decode(out.sequences,skip_special_tokens=True)[0].strip(); conf=0.0
   try:
    ps=[]
    for tok,sc in zip(out.sequences[0][1:],out.scores): ps.append(float(torch.softmax(sc[0],-1)[int(tok)].item()))
    conf=float(np.mean(ps)) if ps else 0
   except Exception: pass
   if conf>best[1]:best=(txt,conf)
  return best
 NUMERIC_KEYS={'ab','d5','d3','cm3','poids','queue','tibia','pied','pouce'}
 def constrained(self,column,path):
  c=column.lower().strip()
  raw,mc=self.raw(path,numeric=c in self.NUMERIC_KEYS)
  if c=='taxon':
   cand=[]
   for x in self.taxons:
    for f,w in [('code',1.15),('nom',1),('scientifique',1)]:cand.append((sim(raw,x[f])*w,x))
   s,x=max(cand,key=lambda z:z[0]); value=x['code']; lc=min(1,s)
  elif c in self.data:
   cand=[(sim(raw,k),k,v) for k,v in self.data[c].items()]; s,k,v=max(cand,key=lambda z:z[0]); value=k; lc=s
  elif c in {'ab','d5','d3','cm3','poids','taille_test','taille_epi','queue','tibia','pied'}:
   m=re.search(r'-?\d+(?:[.,]\d+)?',raw.replace('O','0').replace('o','0')); value=float(m.group().replace(',','.')) if m else raw; lc=.9 if m else .15
  elif c in {'heure','heure capture'}:
   m=re.search(r'\b([01]?\d|2[0-3])\s*[:hH.]\s*([0-5]?\d)\b',raw); value=f'{int(m.group(1)):02d}:{int(m.group(2)):02d}' if m else raw; lc=.9 if m else .15
  else:value=raw;lc=.45 if raw else 0
  conf=.45*mc+.55*lc; return {'raw':raw,'value':value,'confidence':round(conf,3),'level':'fort' if conf>=.75 else ('moyen' if conf>=.5 else 'faible'),'model_confidence':round(mc,3),'lexicon_confidence':round(lc,3)}
