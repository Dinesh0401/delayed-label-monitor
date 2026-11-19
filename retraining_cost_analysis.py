"""Retraining cost-benefit and efficiency analysis across policies and scenarios.
Analyzes trade-offs between refit compute cost, unchanged attempts, and prequential error reduction.
"""
import json
from pathlib import Path

def analyze_cost(summary_path):
    data = json.loads(Path(summary_path).read_text())
    rows = data['rows']

    policies = sorted(list(set(r['policy'] for r in rows)))
    scenarios = sorted(list(set(r['scenario'] for r in rows)))
    sizes = sorted(list(set(r['minority_per_day'] for r in rows)))

    policy_metrics = {p: {
        'total_refits': 0.0,
        'total_promotions': 0.0,
        'total_unchanged': 0.0,
        'total_gain': 0.0,
        'harm_scenarios_count': 0,
        'stable_refits': 0.0
    } for p in policies}

    detailed_table = []

    for r in rows:
        p = r['policy']
        s = r['scenario']
        n = r['minority_per_day']

        refits = r.get('refits', 0.0)
        promotions = r.get('promotions', 0.0)
        unchanged = r.get('unchanged_refits', 0.0)
        gain = r.get('minority_gain_vs_frozen', 0.0)
        wasted_rate = unchanged / max(refits, 1e-6) if refits > 0 else 0.0
        efficiency = gain / max(refits, 1e-6) if refits > 0 else 0.0

        item = {
            'minority_per_day': n,
            'scenario': s,
            'policy': p,
            'minority_error': r.get('minority_error', 0.0),
            'refits': refits,
            'promotions': promotions,
            'unchanged_refits': unchanged,
            'wasted_refit_rate': wasted_rate,
            'minority_gain_vs_frozen': gain,
            'gain_per_refit': efficiency,
            'false_alarms': r.get('false_alarms', 0),
            'detections': r.get('detections', 0)
        }
        detailed_table.append(item)

        policy_metrics[p]['total_refits'] += refits
        policy_metrics[p]['total_promotions'] += promotions
        policy_metrics[p]['total_unchanged'] += unchanged

        if s in ('harm', 'both', 'gradual', 'mild', 'long_tail', 'missing'):
            policy_metrics[p]['total_gain'] += gain
            policy_metrics[p]['harm_scenarios_count'] += 1
        elif s in ('stable', 'reporting_slow', 'reporting_fast'):
            policy_metrics[p]['stable_refits'] += refits

    # Cross-policy summary
    summary = {}
    for p, m in policy_metrics.items():
        total_r = m['total_refits']
        total_u = m['total_unchanged']
        h_count = max(m['harm_scenarios_count'], 1)
        summary[p] = {
            'mean_refits_per_scenario': total_r / len(rows) * len(policies),
            'overall_wasted_refit_rate': total_u / max(total_r, 1e-6) if total_r > 0 else 0.0,
            'mean_harm_gain': m['total_gain'] / h_count,
            'stable_regime_wasted_refits': m['stable_refits'],
            'gain_per_refit': (m['total_gain'] / h_count) / max(total_r / len(rows) * len(policies), 1e-6) if total_r > 0 else 0.0
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

    print("\n--- Cross-Policy Retraining Efficiency Summary ---")
    for p, s in analysis['cross_policy_summary'].items():
        print(f"Policy: {p:15s} | Wasted Refit Rate: {s['overall_wasted_refit_rate']*100:5.1f}% | Stable Wasted Refits: {s['stable_regime_wasted_refits']:5.1f} | Gain/Refit: {s['gain_per_refit']:6.4f}")

if __name__ == '__main__':
    main()
