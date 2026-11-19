"""Pre-registered paired simulations: python research_suite.py (NumPy required)."""
import argparse,csv,gzip,json
from pathlib import Path
import numpy as np
from cautious_monitor import harmful,promotion_bound
DAYS=120; WARM=30; CHANGE=40; WINDOW=28; AGE=9
SCENARIOS=('stable','reporting_slow','reporting_fast','harm','both','mild','gradual','long_tail','missing')
POLICIES=('frozen','periodic','arrived_group','cohort_pooled','cohort_group','ungated_group','shadow_group')

def stream(seed,scenario,minority):
    rng=np.random.default_rng(seed)
    result=[]
    for g,n in enumerate((80,minority)):
        day=np.repeat(np.arange(-WARM,DAYS),n)
        u=rng.random(len(day));v=rng.random(len(day));missing=rng.random(len(day))
        rate=np.full(len(day),.12)
        if g==1:
            if scenario in ('harm','both','long_tail','missing'):rate[day>=CHANGE]=.88
            if scenario=='mild':rate[day>=CHANGE]=.35
            if scenario=='gradual':rate+=.76*np.clip((day-CHANGE)/20,0,1)
        error=u<rate
        early=np.where(error,.3,.8)
        if scenario in ('reporting_slow','both','long_tail','missing'):
            early=np.where(day>=CHANGE,np.where(error,.08,.45),early)
        if scenario=='reporting_fast':early=np.where(day>=CHANGE,np.where(error,.9,.3),early)
        lag=np.where(v<early,1,30 if scenario=='long_tail' else 9)
        arrival=day+lag
        if scenario=='missing':arrival=np.where((day>=CHANGE)&(missing<.2),10**6,arrival)
        result.append((day,error,arrival,n))
    return result

COUNT_CACHE={}

def counts(data,start,end,now,arrived_only=False):
    day,error,arrival,n=data
    key=(id(data),start,end,now,arrived_only)
    if key in COUNT_CACHE:return COUNT_CACHE[key]
    lo=max(0,(start+WARM)*n);hi=max(0,min(len(day),(end+WARM)*n))
    observed=arrival[lo:hi]<=now
    total=int(observed.sum() if arrived_only else hi-lo)
    result=int(error[lo:hi][observed].sum()),0 if arrived_only else int(hi-lo-observed.sum()),total
    COUNT_CACHE[key]=result
    return result

def train_and_gate(data,now,current,guard=True):
    day,error,arrival,n=data
    # Identity predicts x; inversion predicts 1-x. Errors under identity suffice.
    end=now-AGE
    train=(day>=end-28)&(day<end-14)
    if not train.any() or np.any(arrival[train]>now):return None,'incomplete_training',None
    candidate=int(error[train].mean()>.5)
    valid=(day>=end-14)&(day<end);seen=valid&(arrival<=now)
    gain=int(((error[seen]!=current).astype(int)-(error[seen]!=candidate).astype(int)).sum())
    pending=int(valid.sum()-seen.sum())
    lower=promotion_bound(gain,pending,int(valid.sum()),.05/(2*DAYS*2))
    if candidate==current:return None,'unchanged',lower
    if guard and lower<=.02:return None,'insufficient_gain',lower
    return candidate,'promoted',lower

def run(data,policy,scenario,trace=None):
    model=[0,0];attempts=[0,0];promotions=[0,0];loss=[0,0];alarms=[[],[]];unchanged=0
    shadow=[None,None];last_attempt=[-100,-100]
    for t in range(DAYS):
        reference=[counts(d,-WARM,0,t,policy=='arrived_group') for d in data]
        current=[counts(d,t-AGE-WINDOW,t-AGE,t,policy=='arrived_group') for d in data]
        flags=[harmful(reference[g],current[g])['alarm'] for g in range(2)]
        if policy=='cohort_pooled':
            pooled=lambda rows:tuple(sum(r[i] for r in rows) for i in range(3))
            flags=[harmful(pooled(reference),pooled(current),groups=1)['alarm']]*2
        if policy=='frozen':flags=[False,False]
        if policy=='periodic':flags=[t in (40,80)]*2
        for g in range(2):
            if flags[g] and not promotions[g]:alarms[g].append(t)
            if policy=='shadow_group':
                day,error,arrival,n=data[g]
                if shadow[g] is not None:
                    candidate,opened=shadow[g]
                    if t>=opened+14+AGE:
                        valid=(day>=opened)&(day<opened+14);seen=valid&(arrival<=t)
                        gain=int(((error[seen]!=model[g]).astype(int)-(error[seen]!=candidate).astype(int)).sum())
                        lower=promotion_bound(gain,int(valid.sum()-seen.sum()),int(valid.sum()),.05/(2*DAYS*2))
                        if lower>.02:
                            model[g]=candidate;promotions[g]+=1
                            if trace is not None:trace.append(dict(action='promote',group=g,day=t,opened=opened))
                        shadow[g]=None
                if flags[g] and shadow[g] is None and attempts[g]<2 and not promotions[g] and t-last_attempt[g]>=28:
                    train=(day>=t-AGE-14)&(day<t-AGE)
                    if train.any() and np.all(arrival[train]<=t):
                        candidate=int(error[train].mean()>.5);attempts[g]+=1;last_attempt[g]=t
                        if candidate!=model[g]:
                            shadow[g]=(candidate,t)
                            if trace is not None:trace.append(dict(action='train',group=g,day=t))
                        else:unchanged+=1
            # Fixed decisions before prediction: no current/future outcomes used.
            if policy!='shadow_group' and flags[g] and attempts[g]<2 and not promotions[g]:
                candidate,reason,bound=train_and_gate(data[g],t,model[g],policy not in ('periodic','ungated_group','shadow_group'))
                if reason!='incomplete_training':attempts[g]+=1
                unchanged+=reason=='unchanged'
                if candidate is not None:model[g]=candidate;promotions[g]+=1
            day,error,arrival,n=data[g]
            loss[g]+=int(np.sum(error[(t+WARM)*n:(t+WARM+1)*n]!=model[g]))
    pre=any(t<CHANGE for a in alarms for t in a)
    hits=[t for t in alarms[1] if t>=CHANGE]
    null=scenario in ('stable','reporting_slow','reporting_fast')
    return dict(false_alarm=int(bool(any(alarms))) if null else int(pre),
                eligible=int(not null and not pre),detected=int(not null and not pre and bool(hits)),
                capped_delay=(hits[0]-CHANGE if hits else DAYS-CHANGE) if not null and not pre else None,
                majority_error=loss[0]/(DAYS*data[0][3]),minority_error=loss[1]/(DAYS*data[1][3]),
                refits=sum(attempts),promotions=sum(promotions),unchanged_refits=unchanged)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['initial','shadow'],default='shadow')
    args=parser.parse_args()
    first=5000 if args.phase=='initial' else 6000
    policies=POLICIES[:-1] if args.phase=='initial' else POLICIES
    prefix='initial-' if args.phase=='initial' else ''
    out=Path('research/results');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for minority in (2,8,24):
        for scenario in SCENARIOS:
            for seed in range(first,first+100):
                COUNT_CACHE.clear()
                data=stream(seed,scenario,minority)
                for policy in policies:
                    rows.append(dict(minority_per_day=minority,scenario=scenario,seed=seed,policy=policy,**run(data,policy,scenario)))
            print(minority,scenario,flush=True)
    with (out/(prefix+'runs.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    csv_path=out/(prefix+'runs.csv')
    (out/(prefix+'runs.csv.gz')).write_bytes(gzip.compress(csv_path.read_bytes(),mtime=0))
    summary=[]
    for n in (2,8,24):
        for s in SCENARIOS:
            baseline=[r for r in rows if r['minority_per_day']==n and r['scenario']==s and r['policy']=='frozen']
            for p in policies:
                subset=[r for r in rows if r['minority_per_day']==n and r['scenario']==s and r['policy']==p]
                gains=np.array([b['minority_error']-r['minority_error'] for b,r in zip(baseline,subset)])
                rng=np.random.default_rng(71);boot=gains[rng.integers(0,len(gains),(2000,len(gains)))].mean(axis=1)
                summary.append(dict(minority_per_day=n,scenario=s,policy=p,runs=len(subset),
                    false_alarms=sum(r['false_alarm'] for r in subset),eligible=sum(r['eligible'] for r in subset),
                    detections=sum(r['detected'] for r in subset),
                    mean_capped_delay=float(np.mean([r['capped_delay'] for r in subset if r['capped_delay'] is not None])) if any(r['eligible'] for r in subset) else None,
                    **{k:float(np.mean([r[k] for r in subset])) for k in ('majority_error','minority_error','refits','promotions','unchanged_refits')},
                    minority_gain_vs_frozen=float(gains.mean()),paired_gain_bootstrap_95=np.quantile(boot,[.025,.975]).tolist()))
    (out/(prefix+'summary.json')).write_text(json.dumps(dict(protocol='research/PROTOCOL.md',seeds=[first,first+99],rows=summary),indent=2)+'\n')
if __name__=='__main__':main()
