import argparse,json,html
from pathlib import Path
def main():
 ap=argparse.ArgumentParser();ap.add_argument("output");a=ap.parse_args();root=Path(a.output)
 cols=["taxon","sexe","ab","d5","d3","cm3","poids","taille_test","taille_epi","tv","mam","gest","glandes_taille","glandes_couleur","epiph","chin_spot","usure_dents","age_estime"]
 rows=[]
 for p in sorted(root.iterdir()):
  q=p/"decoded.json"
  if p.is_dir() and q.exists():
   for r in json.loads(q.read_text(encoding="utf-8")):
    cells=[]
    for c in cols:
     x=float(r.get(c+"_confidence",0));cl="high" if x>=.75 else ("mid" if x>=.5 else "low"); raw=html.escape(str(r.get(c+"_raw",""))); val=html.escape(str(r.get(c,""))); cells.append(f'<td class="{cl}" title="OCR brut: {raw}">{val}<small>{x:.0%}</small></td>')
    rows.append(f'<tr><td>{html.escape(p.name)}</td><td>{r["ligne"]}</td>'+''.join(cells)+' </tr>')
 css='body{font-family:Arial;margin:18px;background:#eee}table{border-collapse:collapse;background:white;font-size:12px}th,td{border:1px solid #aaa;padding:5px;white-space:nowrap}th{position:sticky;top:0;background:#ddd}.high{background:#b9e6b9}.mid{background:#fff1a8}.low{background:#f5aaaa}small{display:block;font-size:9px;color:#555}'
 page='<!doctype html><meta charset="utf-8"><title>Chiro OCR V6</title><style>'+css+'</style><h1>Chiro OCR V6 — validation</h1><p>Vert ≥75% · Jaune 50–74% · Rouge &lt;50%. Survoler une cellule affiche l’OCR brut.</p><table><tr><th>Fiche</th><th>Ligne</th>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'+''.join(rows)+'</table>'
 (root/'visualisation.html').write_text(page,encoding='utf-8');print((root/'visualisation.html').resolve())
if __name__=='__main__':main()
