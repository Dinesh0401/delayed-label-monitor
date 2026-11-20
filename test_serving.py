import json,os,tempfile,unittest
from pathlib import Path
from fastapi.testclient import TestClient
from serving import app,probability,vector
from datetime import datetime,timezone

class ServingTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.old=os.environ.get('DELIVERY_DB');os.environ['DELIVERY_DB']=self.tmp.name+'/test.sqlite';self.client=TestClient(app)
 def tearDown(self):
  self.tmp.cleanup()
  if self.old is None:os.environ.pop('DELIVERY_DB',None)
  else:os.environ['DELIVERY_DB']=self.old
 def test_sklearn_parity(self):
  for f in json.loads(Path('inference-fixtures.json').read_text()):self.assertAlmostEqual(probability(f['features']),f['expected'],places=12)
 def test_workflow_retry_conflict_and_persistence(self):
  b=dict(id='TEST-1',state='SP',purchased='2026-01-01T10:00:00Z',due='2026-01-10T00:00:00Z')
  r=self.client.post('/api/orders',json=b);self.assertEqual(r.status_code,200)
  self.assertEqual(self.client.post('/api/orders',json=b).json(),r.json())
  self.assertEqual(self.client.post('/api/orders',json={**b,'state':'AM'}).status_code,409)
  m=self.client.get('/api/metrics').json()[0];self.assertEqual((m['pending'],m['lower'],m['upper']),(1,0,1))
  d=dict(id='TEST-1',delivered='2026-01-09T00:00:00Z')
  self.assertEqual(self.client.post('/api/outcomes',json=d).status_code,200)
  self.assertEqual(self.client.post('/api/outcomes',json=d).status_code,200)
  self.assertEqual(self.client.post('/api/outcomes',json={**d,'delivered':'2026-01-11T00:00:00Z'}).status_code,409)
  self.assertEqual(TestClient(app).get('/api/orders').json()[0]['delivered'],d['delivered'].replace('00Z','00.000Z'))
  m=self.client.get('/api/metrics').json()[0];self.assertEqual(m['pending'],0);self.assertEqual(m['lower'],m['upper'])
 def test_invalid_input_and_missing_order(self):
  b=dict(id='TEST-2',state='XX',purchased='2026-01-01T10:00:00Z',due='2026-01-10T00:00:00Z')
  self.assertEqual(self.client.post('/api/orders',json=b).status_code,422)
  self.assertEqual(self.client.post('/api/outcomes',json={'id':'missing','delivered':'2026-01-02T00:00:00Z'}).status_code,404)
 def test_calendar_features(self):
  x=vector('SP',datetime(2026,1,5,12,tzinfo=timezone.utc),datetime(2026,1,6,12,tzinfo=timezone.utc))
  self.assertEqual(x[:4],[1,0,1,12])
 def test_model_reload_mutates_existing_dict_reference(self):
  from candidate_retrain import MODEL as candidate_model
  old_version=candidate_model['version']
  r=self.client.post('/api/model/reload')
  self.assertEqual(r.status_code,200)
  self.assertEqual(candidate_model['version'],old_version)
  from serving import MODEL as serving_model
  self.assertIs(candidate_model,serving_model)

if __name__=='__main__':unittest.main()
