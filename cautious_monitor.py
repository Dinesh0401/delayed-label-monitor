"""Conservative fixed-cohort bounds; no delay-distribution estimate is required."""
import math

def interval(errors, pending, total, alpha):
    if not (0<alpha<1) or min(errors,pending,total)<0 or errors+pending>total:
        raise ValueError('Invalid counts or alpha')
    if not total:return (0.,1.)
    radius=math.sqrt(math.log(1/alpha)/(2*total))
    return max(0.,errors/total-radius),min(1.,(errors+pending)/total+radius)

def harmful(reference,current,alpha=.05,groups=2,looks=120,margin=.05):
    if groups<1 or looks<1:raise ValueError('Positive groups and looks required')
    tail=alpha/(2*groups*looks)
    ref=interval(*reference,tail);cur=interval(*current,tail)
    return dict(alarm=cur[0]>ref[1]+margin,reference_upper=ref[1],current_lower=cur[0],
                current_upper=cur[1],margin=margin)

def promotion_bound(observed_gain,pending,total,alpha):
    # Each paired loss difference is in [-1,1]. Missing values get -1.
    if total<=0 or pending<0 or pending>total or abs(observed_gain)>total-pending or not 0<alpha<1:
        raise ValueError('Invalid paired validation data')
    return max(-1.,(observed_gain-pending)/total-math.sqrt(2*math.log(1/alpha)/total))
