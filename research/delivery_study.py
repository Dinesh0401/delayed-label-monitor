"""Chronological Olist delivery risk and delayed-label monitoring benchmark."""
import argparse
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score
from drift_policy import posterior, reporting_rate

TRAIN_END = pd.Timestamp("2018-01-01")
TEST_START = pd.Timestamp("2018-04-01")
TEST_END = pd.Timestamp("2018-09-01")


def load_orders(path):
    with ZipFile(path) as z:
        orders = pd.read_csv(z.open("olist_orders_dataset.csv"))
        states = pd.read_csv(z.open("olist_customers_dataset.csv"),usecols=["customer_id","customer_state"])
    df = orders.merge(states,on="customer_id",validate="one_to_one")
    df["purchase"] = pd.to_datetime(df.order_purchase_timestamp)
    df["delivery"] = pd.to_datetime(df.order_delivered_customer_date)
    due = pd.to_datetime(df.order_estimated_delivery_date).dt.normalize() + pd.Timedelta(days=1)
    # Evaluate fulfilled orders with interpretable dates. This conditions on eventual fulfillment.
    df = df[(df.order_status=="delivered") & (df.delivery>=df.purchase) & (due>df.purchase)].copy()
    df["late"] = df.delivery >= due.loc[df.index]
    df["promise_days"] = (due.loc[df.index]-df.purchase).dt.total_seconds()/86400
    # Completion confirmation is the operational reporting contract of this replay.
    df["early"] = df.delivery <= df.purchase + pd.Timedelta(days=14)
    df["dow"] = df.purchase.dt.dayofweek
    df["month"] = df.purchase.dt.month
    df["hour"] = df.purchase.dt.hour
    return df.sort_values("purchase").reset_index(drop=True)


def features(df, columns=None):
    x = pd.get_dummies(df[["promise_days","dow","month","hour","customer_state"]],columns=["customer_state"])
    return x if columns is None else x.reindex(columns=columns,fill_value=0)


def summarize(frame, sample, method, probs):
    return dict(sample=sample,method=method,orders=len(frame),late_rate=frame.late.mean(),
                average_precision=average_precision_score(frame.late,probs),
                brier_loss=brier_score_loss(frame.late,probs))


def monitor(cal, test):
    rows=[]
    for (week,region), group in test.groupby([test.purchase.dt.to_period("W"),"region"]):
        if len(group)<10:
            continue
        prior=cal[cal.region==region]
        rates=[reporting_rate(int(prior.loc[prior.error==v,"early"].sum()),sum(prior.error==v)) for v in (True,False)]
        early=group[group.early]
        errors=int(early.error.sum())
        pending=len(group)-len(early)
        corrected,_=posterior(errors,len(early)-errors,pending,*rates)
        rows.append(dict(week=str(week),region=region,n=len(group),pending=pending,truth=group.error.mean(),
                         observed=early.error.mean(),posterior=corrected,
                         lower=errors/len(group),upper=(errors+pending)/len(group)))
    return pd.DataFrame(rows)


def main(path,output):
    df=load_orders(path)
    train=df[(df.purchase<TRAIN_END)&(df.delivery<TRAIN_END)]
    cal=df[(df.purchase>=TRAIN_END)&(df.purchase<TEST_START)&(df.delivery<TEST_START)].copy()
    test=df[(df.purchase>=TEST_START)&(df.purchase<TEST_END)].copy()
    assert train.delivery.max()<TRAIN_END and cal.delivery.max()<TEST_START
    x=features(train)
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=2,random_state=17)
    model.fit(x,train.late)
    pcal=model.predict_proba(features(cal,x.columns))[:,1]
    thresholds=np.arange(.03,.51,.01)
    cutoff=max(thresholds,key=lambda t:f1_score(cal.late,pcal>=t,zero_division=0))
    probs=model.predict_proba(features(test,x.columns))[:,1]
    for frame,p in [(cal,pcal),(test,probs)]:
        frame["error"]=(p>=cutoff)!=frame.late
        frame["region"]=np.where(frame.customer_state=="SP","SP",
                            np.where(frame.customer_state.isin(["AM","PA","AC","AP","RO","RR","TO"]),"North","Other"))
    scores=[]
    for label,mask in [("all",np.ones(len(test),dtype=bool)),("North",test.region=="North")]:
        g=test[mask]
        scores.extend([summarize(g,label,"training prevalence",np.full(len(g),train.late.mean())),
                       summarize(g,label,"gradient boosting",probs[mask])])
    results=monitor(cal,test)
    assert ((results.lower<=results.truth+1e-12)&(results.truth<=results.upper+1e-12)).all()
    output.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(scores).to_csv(output/"forecast.csv",index=False)
    results.to_csv(output/"weekly.csv",index=False)
    paired=results.dropna(subset=["observed"]).copy()
    for col in ["observed","posterior"]:
        paired[col]=(paired[col]-paired.truth).abs()
    summary=paired.groupby("region").agg(cohorts=("n","size"),orders=("n","sum"),observed_mae=("observed","mean"),posterior_mae=("posterior","mean"))
    summary.to_csv(output/"monitor.csv")
    print(f"train={len(train)} calibration={len(cal)} test={len(test)} threshold={cutoff:.2f} cohorts={len(results)}")
    print(pd.DataFrame(scores).to_string(index=False))
    print(summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("archive",type=Path)
    parser.add_argument("--output",type=Path,default=Path("delivery-results"))
    args=parser.parse_args()
    main(args.archive,args.output)
