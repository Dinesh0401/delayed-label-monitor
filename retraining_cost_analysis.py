"""Retraining cost-benefit and efficiency analysis across policies and scenarios.
Analyzes trade-offs between refit compute cost, unchanged attempts, and prequential error reduction.
"""
import json
from pathlib import Path

def analyze_cost(summary_path):
    data = json.loads(Path(summary_path).read_text())
    rows = data['rows']

    policies = sorted(list(set(r['policy'] for r in rows)))
    harm_scenarios = ('harm', 'both', 'gradual', 'mild', 'long_tail', 'missing')
    null_scenarios = ('stable', 'reporting_slow', 'reporting_fast')

    detailed_table = []
    policy_metrics = {p: {
        'all_refits': 0.0,
        'all_promotions': 0.0,
        'all_unchanged': 0.0,
        'all_cells': 0,

        'harm_refits': 0.0,
        'harm_promotions': 0.0,
        'harm_unchanged': 0.0,
        'harm_gain': 0.0,
        'harm_cells': 0,

        'null_refits': 0.0,
        'null_promotions': 0.0,
        'null_unchanged': 0.0,
        'null_cells': 0,
    } for p in policies}

    for r in rows:
        p = r['policy']
        s = r['scenario']
        n = r['minority_per_day']

        refits = r.get('refits', 0.0)
        promotions = r.get('promotions', 0.0)
        unchanged = r.get('unchanged_refits', 0.0)
        gain = r.get('minority_gain_vs_frozen', 0.0)

        promotion_rate = promotions / max(refits, 1e-6) if refits > 0 else 0.0
        unchanged_rate = unchanged / max(refits, 1e-6) if refits > 0 else 0.0
        efficiency = gain / max(refits, 1e-6) if refits > 0 else 0.0

        item = {
            'minority_per_day': n,
            'scenario': s,
            'policy': p,
            'minority_error': r.get('minority_error', 0.0),
            'refits': refits,
            'promotions': promotions,
            'unchanged_refits': unchanged,
            'promotion_rate': promotion_rate,
            'unchanged_rate': unchanged_rate,
            'minority_gain_vs_frozen': gain,
            'gain_per_refit': efficiency,
            'false_alarms': r.get('false_alarms', 0),
            'detections': r.get('detections', 0)
        }
        detailed_table.append(item)

        m = policy_metrics[p]
        m['all_refits'] += refits
        m['all_promotions'] += promotions
        m['all_unchanged'] += unchanged
        m['all_cells'] += 1

        if s in harm_scenarios:
            m['harm_refits'] += refits
            m['harm_promotions'] += promotions
            m['harm_unchanged'] += unchanged
            m['harm_gain'] += gain
            m['harm_cells'] += 1
        elif s in null_scenarios:
            m['null_refits'] += refits
            m['null_promotions'] += promotions
            m['null_unchanged'] += unchanged
            m['null_cells'] += 1

    summary = {}
    for p, m in policy_metrics.items():
        h_refits = m['harm_refits']
        h_prom = m['harm_promotions']
        h_gain = m['harm_gain']
        h_cells = max(m['harm_cells'], 1)

        n_refits = m['null_refits']
        n_cells = max(m['null_cells'], 1)

        summary[p] = {
            # Harmful drift regimes
            'mean_harm_refits_per_scenario': h_refits / h_cells,
            'mean_harm_promotions_per_scenario': h_prom / h_cells,
            'harm_promotion_success_rate': h_prom / max(h_refits, 1e-6) if h_refits > 0 else 0.0,
            'harm_unchanged_refit_rate': m['harm_unchanged'] / max(h_refits, 1e-6) if h_refits > 0 else 0.0,
            'mean_harm_gain_pp': (h_gain / h_cells) * 100,
            'gain_per_harm_refit_pp': (h_gain / max(h_refits, 1e-6)) * 100 if h_refits > 0 else 0.0,

            # Null / stable regimes (unnecessary training waste)
            'null_regime_total_wasted_refits': n_refits,
            'mean_null_refits_per_scenario': n_refits / n_cells,

            # Overall
            'overall_total_refits': m['all_refits'],
            'overall_total_promotions': m['all_promotions'],
            'overall_promotion_rate': m['all_promotions'] / max(m['all_refits'], 1e-6) if m['all_refits'] > 0 else 0.0,
            'overall_unchanged_rate': m['all_unchanged'] / max(m['all_refits'], 1e-6) if m['all_refits'] > 0 else 0.0
        }

    return {
        'source': summary_path,
        'cross_policy_summary': summary,
        'detailed_records': detailed_table
    }

def main():
    root = Path(__file__).parent
    summary_path = root / 'research/results/summary.json'
    analysis = analyze_cost(str(summary_path))

    out_file = root / 'research/results/cost-benefit.json'
    out_file.write_text(json.dumps(analysis, indent=2) + '\n')
    print("Cost-benefit analysis written to", out_file)

    print("\n--- Cross-Policy Retraining Efficiency Summary (Rigorous Demarcation) ---")
    for p, s in analysis['cross_policy_summary'].items():
        print(f"Policy: {p:15s} | Harm Refits: {s['mean_harm_refits_per_scenario']:4.2f} | Harm Prom Success: {s['harm_promotion_success_rate']*100:5.1f}% | Harm Gain/Refit: {s['gain_per_harm_refit_pp']:5.2f} pp | Null Wasted Refits: {s['null_regime_total_wasted_refits']:4.1f}")

if __name__ == '__main__':
    main()
