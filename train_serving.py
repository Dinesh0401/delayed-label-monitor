"""Train a portable logistic baseline for actual server inference."""
import json,sys,hashlib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score,brier_score_loss,f1_score
from research.delivery_study import load_orders,features,TRAIN_END,TEST_START,TEST_END
from pathlib import Path
df=load_orders(sys.argv[1])
train=df[(df.purchase<TRAIN_END)&(df.delivery<TRAIN_END)]
cal=df[(df.purchase>=TRAIN_END)&(df.purchase<TEST_START)&(df.delivery<TEST_START)]
test=df[(df.purchase>=TEST_START)&(df.purchase<TEST_END)]
x=features(train);scaler=StandardScaler().fit(x)
model=LogisticRegression(max_iter=1000,C=1,random_state=17).fit(scaler.transform(x),train.late)
cp=model.predict_proba(scaler.transform(features(cal,x.columns)))[:,1]
threshold=float(max(np.arange(.01,.51,.01),key=lambda t:f1_score(cal.late,cp>=t)))
coef=model.coef_[0]/scaler.scale_;intercept=float(model.intercept_[0]-np.dot(coef,scaler.mean_))
tx=features(test,x.columns)
probs=model.predict_proba(scaler.transform(tx))[:,1]
portable=1/(1+np.exp(-(tx.to_numpy(dtype=float)@coef+intercept)))
assert np.max(np.abs(probs-portable))<1e-12
artifact=dict(version="olist-logistic-v1",features=list(x.columns),coefficients=coef.tolist(),intercept=intercept,threshold=threshold,train_count=len(train),test_count=len(test),
              average_precision=average_precision_score(test.late,probs),brier=brier_score_loss(test.late,probs),test_prevalence=float(test.late.mean()),
              states=sorted(train.customer_state.unique().tolist()),training_cutoff="2018-01-01",test_period="Apr–Aug 2018")
Path(__file__).with_name("serving-model.json").write_text(json.dumps(artifact,indent=2))
fixtures=[dict(features=tx.iloc[i].astype(float).tolist(),expected=float(probs[i])) for i in range(30)]
(Path(__file__).with_name("research") / "fixtures/inference-fixtures.json").write_text(json.dumps(fixtures))
print({k:v for k,v in artifact.items() if k not in ('features','coefficients','states')})
