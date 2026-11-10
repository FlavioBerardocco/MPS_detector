"""
Confidence intervals (bootstrap) and paired comparisons on the test set.

Usage:
    python statistical_analysis.py FOLDER_WITH_CSV

The folder must contain the CSV files saved by the notebook:
  - pred_test_*.csv : one row per test image (run, arch, variant, file, label, prob, pred)
  - seg_test_*.csv  : one row per test mask (run, backbone, idx, dice, iou)

Output (saved in the current working directory):
  - ci_classification.csv       : ensemble metrics with 95% CI (stratified bootstrap)
  - mcnemar_seg_vs_noseg.csv     : paired comparison with vs without segmentation, per architecture
  - ci_segmentation.csv and wilcoxon_segmentation.csv (if the segmentation CSV files are present)

Requires: pip install numpy pandas scipy
"""
import sys, glob, os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

FOLDER = sys.argv[1] if len(sys.argv) > 1 else "."
N_BOOT = 10000
SEED = 42
rng = np.random.default_rng(SEED)


def metrics(y, p):
    tp = np.sum((y == 1) & (p == 1)); tn = np.sum((y == 0) & (p == 0))
    fp = np.sum((y == 0) & (p == 1)); fn = np.sum((y == 1) & (p == 0))
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {'balanced_accuracy': (sens + spec) / 2, 'sensitivity': sens,
            'specificity': spec, 'precision': prec, 'f1': f1}


def bootstrap_indices(y):
    """Stratified bootstrap: resamples with replacement within each class,
    so every resample has the same number of positives and negatives as the test set."""
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for _ in range(N_BOOT):
        yield np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])


def fp(p):
    """Formats a p-value: 4 decimals, or scientific notation if very small."""
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"


def holm(pvals):
    """Holm-Bonferroni correction for multiple comparisons."""
    pvals = np.asarray(pvals, float); m = len(pvals)
    order = np.argsort(pvals); corr = np.empty(m); cum = 0.0
    for rank, i in enumerate(order):
        cum = max(cum, min(1.0, (m - rank) * pvals[i])); corr[i] = cum
    return corr


# ---------------------------------------------------------------
# 1) Classification: 95% CI of the ensemble metrics
# ---------------------------------------------------------------
files = sorted(glob.glob(os.path.join(FOLDER, "pred_test_*.csv")))
if not files:
    sys.exit(f"No pred_test_*.csv files in {FOLDER}")
pred = {}
for f in files:
    d = pd.read_csv(f).sort_values('file').reset_index(drop=True)
    pred[(d.run[0], d.arch[0], d.variant[0])] = d

rows = []
for (run, arch, variant), d in pred.items():
    y, p = d.label.values, d.pred.values
    point = metrics(y, p)
    boot = pd.DataFrame([metrics(y[i], p[i]) for i in bootstrap_indices(y)])
    for m, v in point.items():
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        rows.append(dict(run=run, arch=arch, variant=variant, metric=m, value=round(v, 4),
                         ci95_low=round(lo, 4), ci95_high=round(hi, 4),
                         n_test=len(y), n_pos=int(y.sum()), n_neg=int((y == 0).sum())))
ci = pd.DataFrame(rows)
ci.to_csv("ci_classification.csv", index=False)
print(f"=== CLASSIFICATION: metric (95% CI, stratified bootstrap, {N_BOOT} resamples) ===")
for (run, arch, variant), g in ci.groupby(['run', 'arch', 'variant'], sort=False):
    print(f"\n[{run}] {arch} | {variant}  (test: {g.n_test.iloc[0]} images, "
          f"{g.n_pos.iloc[0]} positive / {g.n_neg.iloc[0]} negative)")
    for _, r in g.iterrows():
        print(f"  {r.metric:<18} {r.value:.3f}  ({r.ci95_low:.3f} – {r.ci95_high:.3f})")

# ---------------------------------------------------------------
# 2) Paired comparison with vs without segmentation (same architecture)
# ---------------------------------------------------------------
rows = []
for (run, arch, variant), d_seg in pred.items():
    if not variant.startswith('seg_') or (run, arch, 'no_seg') not in pred:
        continue
    d_no = pred[(run, arch, 'no_seg')]
    assert list(d_seg.file) == list(d_no.file), "The two pipelines do not have the same test images!"
    y = d_seg.label.values
    ok_seg, ok_no = d_seg.pred.values == y, d_no.pred.values == y
    b = int(np.sum(ok_seg & ~ok_no))     # correct only WITH segmentation
    c = int(np.sum(~ok_seg & ok_no))     # correct only WITHOUT segmentation
    p_mcn = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0   # exact McNemar test
    # Paired bootstrap CI of the difference in balanced accuracy (with - without)
    diff_point = (metrics(y, d_seg.pred.values)['balanced_accuracy']
                  - metrics(y, d_no.pred.values)['balanced_accuracy'])
    diffs = [metrics(y[i], d_seg.pred.values[i])['balanced_accuracy']
             - metrics(y[i], d_no.pred.values[i])['balanced_accuracy'] for i in bootstrap_indices(y)]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    rows.append(dict(run=run, arch=arch, with_seg=variant, without_seg='no_seg', n_test=len(y),
                     both_correct=int(np.sum(ok_seg & ok_no)), both_wrong=int(np.sum(~ok_seg & ~ok_no)),
                     b_only_with_seg=b, c_only_without_seg=c, p_mcnemar=p_mcn,
                     diff_bal_acc=round(diff_point, 4), diff_ci95_low=round(lo, 4), diff_ci95_high=round(hi, 4)))
if rows:
    mc = pd.DataFrame(rows)
    for run, idx in mc.groupby('run').groups.items():
        mc.loc[idx, 'p_holm'] = holm(mc.loc[idx, 'p_mcnemar'])
    mc.to_csv("mcnemar_seg_vs_noseg.csv", index=False)
    print("\n=== WITH vs WITHOUT SEGMENTATION: exact McNemar test (Holm correction) ===")
    for _, r in mc.iterrows():
        print(f"\n[{r.run}] {r.arch}: b (correct only with seg) = {r.b_only_with_seg}, "
              f"c (correct only without seg) = {r.c_only_without_seg}")
        print(f"  p McNemar = {fp(r.p_mcnemar)} | p Holm = {fp(r.p_holm)}")
        print(f"  Balanced accuracy difference (with - without) = {r.diff_bal_acc:+.3f} "
              f"(95% CI: {r.diff_ci95_low:+.3f} – {r.diff_ci95_high:+.3f})")

# ---------------------------------------------------------------
# 3) Segmentation: CI of the mean Dice/IoU and comparison between backbones
# ---------------------------------------------------------------
sfiles = sorted(glob.glob(os.path.join(FOLDER, "seg_test_*.csv")))
if sfiles:
    seg = {}
    for f in sfiles:
        d = pd.read_csv(f).sort_values('idx').reset_index(drop=True)
        seg[(d.run[0], d.backbone[0])] = d
    rows = []
    for (run, bb), d in seg.items():
        for m in ['dice', 'iou']:
            v = d[m].values
            boot = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            rows.append(dict(run=run, backbone=bb, metric=m, mean=round(v.mean(), 4),
                             sd=round(v.std(), 4), ci95_low=round(lo, 4), ci95_high=round(hi, 4), n=len(v)))
    sci = pd.DataFrame(rows); sci.to_csv("ci_segmentation.csv", index=False)
    print("\n=== SEGMENTATION: mean ± SD (95% bootstrap CI of the mean) ===")
    for _, r in sci.iterrows():
        print(f"  [{r.run}] {r.backbone:<16} {r.metric:<5} {r['mean']:.4f} ± {r.sd:.4f}  "
              f"({r.ci95_low:.4f} – {r.ci95_high:.4f})  n={r.n}")
    rows = []
    for run in sorted({k[0] for k in seg}):
        bbs = sorted(bb for r_, bb in seg if r_ == run)
        for i in range(len(bbs)):
            for j in range(i + 1, len(bbs)):
                a, b_ = seg[(run, bbs[i])], seg[(run, bbs[j])]
                for m in ['dice', 'iou']:
                    p = wilcoxon(a[m], b_[m]).pvalue
                    rows.append(dict(run=run, backbone_A=bbs[i], backbone_B=bbs[j], metric=m,
                                     mean_diff=round(a[m].mean() - b_[m].mean(), 4), p_wilcoxon=p))
    if rows:
        wx = pd.DataFrame(rows)
        for (run, m), idx in wx.groupby(['run', 'metric']).groups.items():
            wx.loc[idx, 'p_holm'] = holm(wx.loc[idx, 'p_wilcoxon'])
        wx.to_csv("wilcoxon_segmentation.csv", index=False)
        print("\n=== SEGMENTATION: paired comparisons between backbones (Wilcoxon, Holm correction) ===")
        for _, r in wx.iterrows():
            print(f"  [{r.run}] {r.backbone_A} vs {r.backbone_B} ({r.metric}): "
                  f"diff = {r.mean_diff:+.4f}, p = {fp(r.p_wilcoxon)}, p Holm = {fp(r.p_holm)}")
print("\nFiles saved in the current directory.")
