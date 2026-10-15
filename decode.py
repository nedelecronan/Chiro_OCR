import argparse,json,csv
from pathlib import Path
import settings
from handwriting_ocr import LocalHandwritingOCR
def main():
 ap=argparse.ArgumentParser();ap.add_argument('output');ap.add_argument('--model',default='models/trocr-handwritten');ap.add_argument('--region',default=settings.region.name,choices=settings.REGIONS,help='Référentiel de taxons à utiliser');a=ap.parse_args();settings.region.set_region(a.region);o=LocalHandwritingOCR(a.model);o.load();root=Path(a.output);allrows=[]
 for p in sorted(root.iterdir()):
  if not p.is_dir() or not (p/'manifest.json').exists():continue
  man=json.loads((p/'manifest.json').read_text(encoding='utf-8'));rows={}
  for it in man:
   img_path=p/f"ligne_{int(it['ligne']):02d}"/Path(it['clean']).name
   rows.setdefault(int(it['ligne']),{})[it['colonne']]=o.constrained(it['colonne'],img_path)
  out=[]
  for n,d in sorted(rows.items()):
   r={'ligne':n}
   for c,v in d.items():r[c]=v['value'];r[c+'_raw']=v['raw'];r[c+'_confidence']=v['confidence'];r[c+'_level']=v['level']
   out.append(r);allrows.append({'fiche':p.name,**r})
  (p/'decoded.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 cols=[]
 for r in allrows:
  for k in r:
   if k not in cols:cols.append(k)
 with open(root/'donnees_decodees.csv','w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=cols,delimiter=';');w.writeheader();w.writerows(allrows)
 print(root/'donnees_decodees.csv')
if __name__=='__main__':main()
