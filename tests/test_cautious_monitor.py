import unittest
import numpy as np
from cautious_monitor import interval,harmful,promotion_bound
from research_suite import stream,counts,train_and_gate,COUNT_CACHE

class CautiousTests(unittest.TestCase):
 def test_pending_outcomes_cannot_create_error_evidence(self):
  self.assertEqual(interval(0,0,0,.01),(0,1))
  self.assertLessEqual(interval(5,95,100,.01)[0],.05)
  self.assertFalse(harmful((12,0,100),(0,100,100))['alarm'])
 def test_exact_completion_sandwich(self):
  for k in range(6):
   lower,upper=interval(2,5,10,.1)
   self.assertLessEqual(lower,(2+k)/10)
   self.assertGreaterEqual(upper,(2+k)/10)
 def test_strong_harm_and_no_change(self):
  self.assertTrue(harmful((120,0,1000),(880,0,1000))['alarm'])
  self.assertFalse(harmful((120,0,1000),(120,0,1000))['alarm'])
 def test_promotion_rejects_sparse_or_missing_validation(self):
  self.assertLess(promotion_bound(5,0,5,.001),.02)
  self.assertLess(promotion_bound(0,100,100,.001),0)
  self.assertGreater(promotion_bound(800,0,1000,.001),.02)
 def test_no_future_label_leakage(self):
  data=stream(99999,'both',8)[1];now=65
  COUNT_CACHE.clear();before=counts(data,28,56,now)
  altered=tuple(x.copy() if isinstance(x,np.ndarray) else x for x in data)
  altered[1][altered[2]>now]=~altered[1][altered[2]>now]
  COUNT_CACHE.clear();self.assertEqual(before,counts(altered,28,56,now))
  self.assertEqual(train_and_gate(data,now,0),train_and_gate(altered,now,0))
 def test_fixed_cohort_counts(self):
  data=stream(99999,'stable',8)[1];COUNT_CACHE.clear()
  errors,pending,total=counts(data,10,20,20)
  mask=(data[0]>=10)&(data[0]<20)
  self.assertEqual(total,80)
  self.assertEqual(errors,int(data[1][mask&(data[2]<=20)].sum()))
  self.assertEqual(pending,int((mask&(data[2]>20)).sum()))
 def test_invalid_counts(self):
  with self.assertRaises(ValueError):interval(10,1,10,.05)

if __name__=='__main__':unittest.main()

class OperationalTests(unittest.TestCase):
 def test_new_data_is_not_drift_evidence(self):
  from operational_diagnostics import diagnose
  from datetime import datetime,timezone
  row=dict(state='SP',model_version='v1',created='2026-01-01T10:00:00Z',reported=None)
  result=diagnose([row],datetime(2026,1,2,tzinfo=timezone.utc))
  self.assertEqual(result['groups'][0]['status'],'collecting_reference')
 def test_daily_snapshot_and_mature_reference(self):
  from operational_diagnostics import diagnose
  from datetime import datetime,timedelta,timezone
  start=datetime(2026,1,1,tzinfo=timezone.utc);rows=[]
  for day in range(100):
   for i in range(100):
    d=start+timedelta(days=day,hours=1);s=d.isoformat()
    rows.append(dict(state='SP',model_version='v1',created=s,reported=s,delivered=s,
                     due=(d+timedelta(days=1 if day<40 else -1)).isoformat(),predicted=0))
  result=diagnose(rows,start+timedelta(days=100))
  self.assertEqual(result['groups'][0]['status'],'harm_evidence')


class ShadowTests(unittest.TestCase):
 def test_prospective_validation_precedes_promotion(self):
  from research_suite import run
  data=stream(6000,'both',8);COUNT_CACHE.clear();trace=[]
  result=run(data,'shadow_group','both',trace)
  self.assertEqual(result['promotions'],1)
  for event in trace:
   if event['action']=='promote':self.assertGreaterEqual(event['day'],event['opened']+23)
 def test_future_errors_do_not_change_past_decisions(self):
  from research_suite import run
  data=stream(6000,'both',8);COUNT_CACHE.clear();before=[]
  run(data,'shadow_group','both',before)
  altered=[tuple(x.copy() if isinstance(x,np.ndarray) else x for x in d) for d in data]
  for d in altered:d[1][d[2]>65]=~d[1][d[2]>65]
  COUNT_CACHE.clear();after=[];run(altered,'shadow_group','both',after)
  self.assertEqual([e for e in before if e['day']<=65],[e for e in after if e['day']<=65])
