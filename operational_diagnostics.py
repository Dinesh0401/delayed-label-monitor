"""One daily look; cohorts use prediction creation time, never label arrival selection."""
from datetime import datetime,timezone
from cautious_monitor import harmful
DAY=86400

def epoch(value):return datetime.fromisoformat(value.replace('Z','+00:00')).timestamp()

def diagnose(rows,now=None):
    cutoff=int((now or datetime.now(timezone.utc)).timestamp()//DAY)*DAY
    groups={}
    for row in rows:
        if epoch(row['created'])<cutoff:
            groups.setdefault((row['state'],row['model_version']),[]).append(row)
    output=[]
    for (state,version),events in sorted(groups.items()):
        start=int(min(epoch(r['created']) for r in events)//DAY)*DAY
        day=int((cutoff-start)//DAY)
        base=dict(state=state,model_version=version,day=day,as_of=datetime.fromtimestamp(cutoff,timezone.utc).isoformat())
        if day<67:
            output.append(dict(**base,status='collecting_reference',days_until_first_look=67-day));continue
        if day>=150:
            output.append(dict(**base,status='horizon_complete'));continue
        def cohort(lo,hi):
            chosen=[r for r in events if start+lo*DAY<=epoch(r['created'])<start+hi*DAY]
            known=[r for r in chosen if r['reported'] and epoch(r['reported'])<=cutoff]
            wrong=sum(bool(r['predicted'])!=(epoch(r['delivered'])>=epoch(r['due'])) for r in known)
            return wrong,len(chosen)-len(known),len(chosen)
        reference=cohort(0,30);current=cohort(day-37,day-9)
        if not reference[2] or not current[2]:
            output.append(dict(**base,status='insufficient_data'));continue
        result=harmful(reference,current,groups=27,looks=120)
        output.append(dict(**base,**result,reference_n=reference[2],current_n=current[2],pending=current[1],
                           status='harm_evidence' if result['alarm'] else 'no_confirmed_harm'))
    return dict(as_of=datetime.fromtimestamp(cutoff,timezone.utc).isoformat(),groups=output,
                policy='Daily fixed cohorts, 5 percentage-point harm margin; no automatic promotion',
                assumptions='Independent bounded losses, fixed model, fixed 27 states; one 120-look budget per model version. Reusing versions or repeated restarts invalidates the budget.')
