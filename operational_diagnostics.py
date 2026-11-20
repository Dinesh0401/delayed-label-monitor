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
        # Multiscale evaluation: when day >= 88, also evaluate matured cohort (30-day buffer)
        is_multiscale = day >= 88
        current_mature = cohort(day - 58, day - 30) if is_multiscale else None
        if is_multiscale and current_mature and current_mature[2] > 0:
            from matured_monitor import multiscale_harmful
            ms = multiscale_harmful(reference, current, reference, current_mature, groups=27, looks=120)
            alarm = ms['alarm']
            scale = ms['scale']
            ref_upper = min(ms['short_ref_upper'], ms['mature_ref_upper'])
            cur_lower = max(ms['short_cur_lower'], ms['mature_cur_lower'])
        else:
            res = harmful(reference, current, groups=27, looks=120)
            alarm = res['alarm']
            scale = 'short' if alarm else 'none'
            ref_upper = res['reference_upper']
            cur_lower = res['current_lower']
        output.append(dict(**base, alarm=alarm, status='harm_evidence' if alarm else 'no_confirmed_harm',
                           scale_triggered=scale, multiscale_active=is_multiscale,
                           reference_upper=ref_upper, current_lower=cur_lower,
                           reference_n=reference[2], current_n=current[2], pending=current[1], margin=.05))
    return dict(as_of=datetime.fromtimestamp(cutoff,timezone.utc).isoformat(),groups=output,
                policy='Daily multiscale fixed cohorts, 5 percentage-point harm margin; no automatic promotion',
                assumptions='Independent bounded losses, fixed model, fixed 27 states; one 120-look budget per model version. Reusing versions or repeated restarts invalidates the budget.')
