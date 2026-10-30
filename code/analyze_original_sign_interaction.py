"""Post-analysis paired interaction of fixed LAD forecast gains within origin-sign groups."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/original-forecast/sign_interaction_diagnostic'
DESIGN = ROOT / 'config/original-sign-interaction.txt'
PRED = ROOT / 'analysis/original-forecast/nesting_order_diagnostic/predictions.csv'
OLD = ROOT / 'analysis/original-forecast/fixed_forecast_decomposition'
CELLS = ['lad_ar', 'lad_linear_signals', 'lad_piecewise', 'lad_signals']
WEIGHTS = {
    'accounting_first': np.array([1., -1., 0., 0.]),
    'asymmetry_second': np.array([0., 1., 0., -1.]),
    'asymmetry_first': np.array([1., 0., -1., 0.]),
    'accounting_second': np.array([0., 0., 1., -1.]),
}
INTERACTION = np.array([-1., 1., 1., -1.])
SEED, DRAWS = 20261025, 2000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(np.asarray(array, dtype='<f8').tobytes(order='C')).hexdigest()


def main():
    assert DESIGN.exists()
    started = datetime.now(timezone.utc).isoformat()
    paths = [PRED, OLD/'group_metrics.csv', OLD/'group_contrasts.csv']
    original_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    code_sha, design_sha = sha(Path(__file__)), sha(DESIGN)
    pred = pd.read_csv(PRED, dtype={'stock_code': str})
    p = pred[pred.target_year.between(2015, 2019)].copy()
    assert len(p) == 15258 and not p.duplicated(['stock_code', 'year']).any()
    assert p.target_year.eq(p.year+1).all()
    assert np.isfinite(p[['roe_pct', 'target']+CELLS]).all().all()
    assert np.array_equal(p.negative_roe.to_numpy(), p.roe_pct.lt(0).astype(int).to_numpy())
    old_metrics = pd.read_csv(OLD/'group_metrics.csv')
    old_contrasts = pd.read_csv(OLD/'group_contrasts.csv')
    metrics, component_checks, interactions = [], [], []
    draw_hashes, numerical_checks = {}, {}
    for name, mask in [('negative', p.roe_pct.lt(0)), ('nonnegative', p.roe_pct.ge(0))]:
        frame = p[mask]
        losses = frame[CELLS].subtract(frame.target, axis=0).abs()
        means = losses.mean().to_numpy()
        n, clusters = len(frame), frame.stock_code.nunique()
        existing_m = old_metrics[old_metrics.group_scope.eq('origin_sign') &
            old_metrics.provenance.eq('all') & old_metrics.origin_roe_group.eq(name)]
        assert len(existing_m) == 1
        existing_m = existing_m.iloc[0]
        assert existing_m.n == n and existing_m.securities == clusters
        np.testing.assert_allclose(means, existing_m[[c+'_mae' for c in CELLS]].to_numpy(float), rtol=0, atol=1e-10)
        metrics.append({'origin_roe_group': name, 'n': n, 'securities': clusters,
            **{c+'_mae_pp': float(v) for c,v in zip(CELLS, means)}})
        # All four losses use the same sampled securities in each bootstrap draw.
        aggregate = losses.copy()
        aggregate['stock_code'], aggregate['n'] = frame.stock_code, 1
        values = aggregate.groupby('stock_code', sort=True)[CELLS+['n']].sum().to_numpy()
        assert len(values) == clusters and values[:,4].sum() == n
        rng = np.random.default_rng(SEED)
        joint_means = np.empty((DRAWS, 4))
        for draw in range(DRAWS):
            total = values[rng.integers(0, clusters, clusters)].sum(axis=0)
            joint_means[draw] = total[:4]/total[4]
        component_draws, component_points = {}, {}
        max_existing_difference = 0.
        for contrast, weight in WEIGHTS.items():
            point = float(means@weight)
            draws = joint_means@weight
            lower, upper = np.quantile(draws, [.025, .975])
            prior = old_contrasts[old_contrasts.group_scope.eq('origin_sign') &
                old_contrasts.provenance.eq('all') & old_contrasts.origin_roe_group.eq(name) &
                old_contrasts.comparison.eq(contrast)]
            assert len(prior) == 1
            prior = prior.iloc[0]
            assert prior.n == n and prior.clusters == clusters and prior.seed == SEED and prior.bootstrap_draws == DRAWS
            new_values = np.array([point, lower, upper])
            old_values = prior[['mae_reduction_pp','ci_low_pp','ci_high_pp']].to_numpy(float)
            np.testing.assert_allclose(new_values, old_values, rtol=0, atol=1e-10)
            discrepancy = float(np.max(abs(new_values-old_values)))
            max_existing_difference = max(max_existing_difference, discrepancy)
            component_checks.append({'origin_roe_group': name, 'comparison': contrast,
                'n': n, 'clusters': clusters, 'mae_reduction_pp': point,
                'ci_low_pp': float(lower), 'ci_high_pp': float(upper),
                'max_absolute_difference_from_existing': discrepancy})
            component_draws[contrast], component_points[contrast] = draws, point
        effect = float(means@INTERACTION)
        draws = joint_means@INTERACTION
        lower, upper = np.quantile(draws, [.025, .975])
        accounting_difference = component_draws['accounting_second']-component_draws['accounting_first']
        asymmetry_difference = component_draws['asymmetry_second']-component_draws['asymmetry_first']
        path1 = component_draws['accounting_first']+component_draws['asymmetry_second']
        path2 = component_draws['asymmetry_first']+component_draws['accounting_second']
        np.testing.assert_allclose(draws, accounting_difference, rtol=0, atol=1e-10)
        np.testing.assert_allclose(draws, asymmetry_difference, rtol=0, atol=1e-10)
        np.testing.assert_allclose(path1, path2, rtol=0, atol=1e-10)
        np.testing.assert_allclose(effect, (losses.to_numpy()@INTERACTION).mean(), rtol=0, atol=1e-10)
        np.testing.assert_allclose(effect, component_points['accounting_second']-component_points['accounting_first'], rtol=0, atol=1e-10)
        interactions.append({'origin_roe_group': name, 'n': n, 'clusters': clusters,
            'accounting_first_gain_pp': component_points['accounting_first'],
            'accounting_second_gain_pp': component_points['accounting_second'],
            'accounting_second_minus_first_pp': effect,
            'ci_low_pp': float(lower), 'ci_high_pp': float(upper),
            'bootstrap_draws': DRAWS, 'seed': SEED})
        draw_hashes[name] = {'four_joint_mae_draws_float64_le_sha256': array_sha(joint_means),
            'interaction_draws_float64_le_sha256': array_sha(draws)}
        numerical_checks[name] = {'max_existing_component_difference': max_existing_difference,
            'accounting_identity_max_error': float(abs(draws-accounting_difference).max()),
            'asymmetry_identity_max_error': float(abs(draws-asymmetry_difference).max()),
            'nesting_path_sum_max_error': float(abs(path1-path2).max())}
    assert sum(r['n'] for r in interactions) == len(p)
    pooled = p[CELLS].subtract(p.target, axis=0).abs().mean().to_numpy()@INTERACTION
    weighted = sum(r['n']*r['accounting_second_minus_first_pp'] for r in interactions)/len(p)
    np.testing.assert_allclose(pooled, weighted, rtol=0, atol=1e-10)
    assert original_hashes == {str(p.relative_to(ROOT)):sha(p) for p in paths}
    assert code_sha == sha(Path(__file__)) and design_sha == sha(DESIGN)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(OUT/'group_maes.csv', index=False)
    pd.DataFrame(component_checks).to_csv(OUT/'existing_component_checks.csv', index=False)
    pd.DataFrame(interactions).to_csv(OUT/'interaction_intervals.csv', index=False)
    summary = {'status':'post-analysis fixed-forecast sign interaction; no refits',
        'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'code_sha256': code_sha, 'design_sha256': design_sha, 'input_sha256': original_hashes,
        'inputs_unchanged': True, 'cells': CELLS, 'interaction_weights': INTERACTION.tolist(),
        'estimand':'Accounting-second minus accounting-first within each raw-origin-ROE sign group; also asymmetry-second minus asymmetry-first.',
        'bootstrap':'Within-sign sorted security clusters; numpy default_rng seed reset for each group; 2000 joint draws; percentile 95% interval.',
        'conditional_scope':'Fixed forecasts and realized calendar years; no refitting, between-sign-group difference test, causal coefficient interpretation, or multiplicity adjustment.',
        'draw_fingerprints': draw_hashes, 'checks': numerical_checks,
        'pooled_weighted_point_identity_error':float(abs(pooled-weighted)),
        'existing_component_intervals_checked':len(component_checks),
        'row_level_data_distributed':False, 'primary_outputs_modified':False}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(pd.DataFrame(interactions).to_string(index=False))
    print(json.dumps(numerical_checks, indent=2))


if __name__ == '__main__':
    main()
