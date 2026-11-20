import json
import os
import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from fastapi.testclient import TestClient

from serving import app
from auto_pipeline import ingest_from_dict, check_stale_orders, run_gated_promotion

class AutoPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp.name, 'test_pipeline.sqlite')
        self.old_db = os.environ.get('DELIVERY_DB')
        os.environ['DELIVERY_DB'] = self.db_path
        self.client = TestClient(app)

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except Exception:
            pass
        if self.old_db is None:
            os.environ.pop('DELIVERY_DB', None)
        else:
            os.environ['DELIVERY_DB'] = self.old_db

    def test_batch_orders_and_outcomes_api(self):
        orders = [
            dict(id='BATCH-1', state='SP', purchased='2026-01-01T10:00:00Z', due='2026-01-10T00:00:00Z'),
            dict(id='BATCH-2', state='RJ', purchased='2026-01-02T10:00:00Z', due='2026-01-12T00:00:00Z')
        ]
        r = self.client.post('/api/batch-orders', json=orders)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()['inserted'], 2)

        # Batch outcomes - reject future delivery dates (e.g. year 2099)
        outcomes_invalid = [
            dict(id='BATCH-2', delivered='2099-01-01T00:00:00Z')
        ]
        r_out_inv = self.client.post('/api/batch-outcomes', json=outcomes_invalid)
        self.assertEqual(r_out_inv.status_code, 200)
        self.assertEqual(r_out_inv.json()['updated'], 0)  # Future date must be rejected

        # Valid batch outcome
        outcomes = [
            dict(id='BATCH-1', delivered='2026-01-08T12:00:00Z')
        ]
        r_out = self.client.post('/api/batch-outcomes', json=outcomes)
        self.assertEqual(r_out.status_code, 200)
        self.assertEqual(r_out.json()['updated'], 1)

        # Export endpoint
        r_exp = self.client.get('/api/export')
        self.assertEqual(r_exp.status_code, 200)
        self.assertEqual(r_exp.json()['total'], 2)

        # Pipeline status endpoint
        r_stat = self.client.get('/api/pipeline-status')
        self.assertEqual(r_stat.status_code, 200)
        stat = r_stat.json()
        self.assertEqual(stat['total_orders'], 2)
        self.assertEqual(stat['resolved_orders'], 1)
        self.assertEqual(stat['pending_orders'], 1)

    def test_stale_orders_check(self):
        past_date = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat().replace('+00:00', 'Z')
        past_due = (datetime.now(timezone.utc) - timedelta(days=16)).isoformat().replace('+00:00', 'Z')
        payload = {
            'orders': [
                dict(id='STALE-1', state='MG', purchased=past_date, due=past_due)
            ]
        }
        ingest_from_dict(payload, self.db_path)
        stale = check_stale_orders(self.db_path, stale_days=14)
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]['id'], 'STALE-1')

    def test_gated_promotion_staging_leaves_live_model_untouched(self):
        # Generate synthetic orders for candidate retraining test
        records = []
        for i in range(240):
            train = i < 120
            days = 3 if i % 2 else 20
            purchased = '2026-01-01T00:00:00Z' if train else '2026-03-01T00:00:00Z'
            month = '01' if train else '03'
            records.append(dict(
                id=str(i), state='SP' if i % 3 else 'AM',
                purchased=purchased,
                due=f'2026-{month}-{days+1:02d}T00:00:00Z',
                created=purchased,
                delivered=f'2026-{month}-10T00:00:00Z',
                reported=f'2026-{month}-11T00:00:00Z'
            ))

        input_file = Path(self.tmp.name) / 'export.json'
        input_file.write_text(json.dumps({'orders': records, 'scope': 'test_staging'}))

        stage_dir = Path(self.tmp.name) / 'staged'
        live_model_before = Path('serving-model.json').read_text()

        report = run_gated_promotion(
            str(input_file),
            '2026-04-01T00:00:00Z',
            '2026-02-01T00:00:00Z',
            stage_dir=str(stage_dir),
            confirm_replace=False
        )

        # Staged files exist
        self.assertTrue((stage_dir / 'candidate-model.json').exists())
        self.assertTrue((stage_dir / 'promotion-report.json').exists())
        self.assertEqual(report['confirm_replace_flag'], False)

        # Crucial safety check: live model was NOT modified
        live_model_after = Path('serving-model.json').read_text()
        self.assertEqual(live_model_before, live_model_after)

if __name__ == '__main__':
    unittest.main()
