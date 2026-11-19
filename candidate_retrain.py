"""Train a shadow model from exported orders; never overwrite the deployed model.
Example: python candidate_retrain.py export.json --as-of 2026-11-19T00:00:00Z
         --train-before 2026-10-01T00:00:00Z --out candidate-run
Run once on a held-out validation period; repeated tuning invalidates its gate.
"""
import argparse,hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from cautious_monitor import promotion_bound
from serving import MODEL,vector,probability

def time(value):
    d=datetime.fromisoformat(value.replace('Z','+00:00'))
    if d.tzinfo is None:raise ValueError('Timestamps need a timezone')
    return d.astimezone(timezone.utc)

def evaluate_gate(old,new,labels,groups,alpha=.05,cost=.005,tolerance=.02):
    groups=np.asarray(groups);labels=np.asarray(labels);old=np.asarray(old);new=np.asarray(new)
    names=sorted(set(groups));rows=[]
    for name in ['ALL',*names]:
        mask=np.ones(len(labels),dtype=bool) if name=='ALL' else groups==name
        known=mask&(labels>=0)
        gain=int(((old[known]!=labels[known]).astype(int)-(new[known]!=labels[known]).astype(int)).sum())
        n=int(mask.sum());pending=n-int(known.sum())
        lower=promotion_bound(gain,pending,n,alpha/(len(names)+1))
        rows.append(dict(group=name,total=n,pending=pending,lower_gain=lower,
                         required_gain=cost if name=='ALL' else -tolerance,
                         passes=lower>(cost if name=='ALL' else -tolerance)))
    return dict(accepted=all(r['passes'] for r in rows),groups=rows,
                scope='One pre-specified candidate and held-out split; not a perpetual guarantee')

def retrain(source,asof,cut,out):
    if cut>=asof:raise ValueError('Training cutoff must precede evaluation time')
    payload=json.loads(Path(source).read_text());orders=payload['orders']
    if len({r['id'] for r in orders})!=len(orders):raise ValueError('Duplicate order IDs')
    eligible=[r for r in orders if time(r['purchased'])<asof and time(r['created'])<=asof]
    train=[r for r in eligible if time(r['purchased'])<cut and r.get('reported') and time(r['reported'])<=asof]
    valid=[r for r in eligible if time(r['purchased'])>=cut]
    if len(train)<30 or not valid:raise ValueError('Need >=30 labeled training orders and a nonempty later validation cohort')
    features=lambda rows:np.asarray([vector(r['state'],time(r['purchased']),time(r['due'])) for r in rows])
    labels=lambda rows:np.asarray([int(time(r['delivered'])>=time(r['due'])) if r.get('reported') and time(r['reported'])<=asof else -1 for r in rows])
    x=features(train);y=labels(train)
    if len(set(y))!=2:raise ValueError('Training requires both on-time and late outcomes')
    scaler=StandardScaler().fit(x)
    fitted=LogisticRegression(C=1,max_iter=1000,random_state=17).fit(scaler.transform(x),y)
    coef=fitted.coef_[0]/scaler.scale_;intercept=float(fitted.intercept_[0]-coef@scaler.mean_)
    vx=features(valid);cp=1/(1+np.exp(-np.clip(vx@coef+intercept,-700,700)))
    assert np.max(np.abs(cp-fitted.predict_proba(scaler.transform(vx))[:,1]))<1e-12
    baseline=np.asarray([probability(row) for row in vx])>=MODEL['threshold']
    candidate=cp>=MODEL['threshold']
    report=evaluate_gate(baseline,candidate,labels(valid),[r['state'] for r in valid])
    report.update(training_orders=len(train),validation_orders=len(valid),as_of=asof.isoformat(),
                  train_before=cut.isoformat(),source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  deployment='No deployed artifact was changed',export_scope=payload.get('scope','unspecified'))
    artifact={**MODEL,'version':'shadow-'+report['source_sha256'][:12],'coefficients':coef.tolist(),'intercept':intercept,
              'train_count':len(train),'training_cutoff':cut.isoformat()}
    # Old historical metrics do not describe this newly trained model.
    for k in ('test_count','average_precision','brier','test_prevalence','test_period'):artifact.pop(k,None)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'candidate-model.json').write_text(json.dumps(artifact,indent=2)+'\n')
    (out/'promotion-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input');p.add_argument('--as-of',required=True);p.add_argument('--train-before',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    print(json.dumps(retrain(a.input,time(a.as_of),time(a.train_before),a.out),indent=2))
