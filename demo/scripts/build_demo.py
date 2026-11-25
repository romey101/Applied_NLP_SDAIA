import csv, json, math
from pathlib import Path
from collections import Counter
root = Path(__file__).resolve().parents[2]
source = root / 'data/processed'
def read(name):
    with (source / (name + '.csv')).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))
points = []
for r in read('clusters'):
    x,y = float(r['x']),float(r['y'])
    label,cluster = int(float(r['label'])),int(float(r['cluster']))
    if not (math.isfinite(x) and math.isfinite(y) and label in (0,1)):
        raise ValueError('Invalid cluster row')
    points.append(dict(review_id=len(points)+1,x=x,y=y,label=label,cluster=cluster))
counts=Counter(p['cluster'] for p in points)
positive=Counter(p['cluster'] for p in points if p['label']==1)
topics=[dict(cluster=k,name_en=f'Topic {k+1}',name_ar=f'الموضوع {k+1}',count=n,positive_rate=positive[k]/n,keywords=[]) for k,n in sorted(counts.items())]
# Explicit allowlists: never copy review text or arbitrary CSV columns.
allowed={'final_sentiment_metrics':{'model','accuracy','macro_precision','macro_recall','macro_f1','ci_low','ci_high'},'cluster_summary':{'k','silhouette','davies_bouldin','seed_stability_ari'},'cluster_metrics':{'k','silhouette','davies_bouldin','seed_stability_ari'},'ner_summary':{'reviews','entities','filtered','coverage'}}
def scalar(v):
    try:
        n=float(v)
        return n if math.isfinite(n) else None
    except (ValueError,TypeError): return None
metrics={}
for name, cols in allowed.items():
    if (source/(name+'.csv')).exists():
        metrics[name]=[{k:(v if k=='model' and v in ('baseline','distilbert','pred_baseline','pred_distilbert','baseline_pred','distilbert_pred') else scalar(v)) for k,v in r.items() if k in cols} for r in read(name)]
metrics['sentiment_metrics']=metrics.get('final_sentiment_metrics',[])
result=dict(overview=dict(total=len(points),positive=sum(positive.values())),universe=dict(points=points,topics=topics),entities=metrics.get('ner_summary',[]),metrics=metrics,health=dict(root='Public demo — static exports',artifacts={'demo.json':True},models={}))
out=root/'demo/public/demo.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(f'Exported {len(points)} indexed sample points')
