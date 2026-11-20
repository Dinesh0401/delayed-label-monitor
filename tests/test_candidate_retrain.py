import unittest,json,tempfile
from pathlib import Path
from candidate_retrain import evaluate_gate,retrain,time

class CandidateTests(unittest.TestCase):
 def test_small_group_blocks_global_gain(self):
  old=[0]*2000+[1]*2;new=[1]*2000+[0]*2;labels=[1]*2002
  result=evaluate_gate(old,new,labels,['majority']*2000+['minority']*2)
  self.assertFalse(result['accepted'])
  self.assertTrue(result['groups'][0]['passes'])
 def test_missing_labels_cannot_authorize_promotion(self):
  self.assertFalse(evaluate_gate([0]*100,[1]*100,[-1]*100,['A']*100)['accepted'])
 def test_clear_group_gain_can_pass(self):
  self.assertTrue(evaluate_gate([0]*2000,[1]*2000,[1]*2000,['A']*1000+['B']*1000)['accepted'])
 def test_actual_training_pipeline(self):
  records=[]
  for i in range(240):
   train=i<120;days=3 if i%2 else 20
   purchased='2026-01-01T00:00:00Z' if train else '2026-03-01T00:00:00Z'
   month='01' if train else '03'
   records.append(dict(id=str(i),state='SP' if i%3 else 'AM',purchased=purchased,
     due=f'2026-{month}-{days+1:02d}T00:00:00Z',created=purchased,
     delivered=f'2026-{month}-10T00:00:00Z',reported=f'2026-{month}-11T00:00:00Z'))
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/'input.json';source.write_text(json.dumps({'orders':records,'scope':'synthetic test fixture'}))
   r=retrain(source,time('2026-04-01T00:00:00Z'),time('2026-02-01T00:00:00Z'),Path(d)/'out')
   self.assertEqual(r['training_orders'],120);self.assertEqual(r['validation_orders'],120)
   self.assertTrue((Path(d)/'out/candidate-model.json').exists())
   self.assertEqual(r['deployment'],'No deployed artifact was changed')

if __name__=='__main__':unittest.main()
