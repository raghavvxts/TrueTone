"""Paired grouped evaluation: same encoders/classifier/test clips, different training data."""
import json
import hashlib
import os
from pathlib import Path
import re
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from inference import fit_bundle

warnings.filterwarnings('ignore', category=FutureWarning)
DATA = ROOT / 'truetone/pp'
OUT = ROOT / 'results/mustardpp'
OUT.mkdir(parents=True, exist_ok=True)
MODES = {'Text only': ['text_emb', 'text_sent'], 'Speech reference': ['wavlm', 'prosody', 'speech_emo'],
         'Text + speech': ['text_emb', 'text_sent', 'wavlm', 'prosody', 'speech_emo', 'mismatch']}


def normalize(text):
    return re.sub(r'[^a-z0-9]', '', text.lower())


def leakage_groups(rows, baseline, ids):
    """Union shared IDs, identical target text and identical decoded audio."""
    records = [(key, baseline[key]['utterance'], ROOT/'truetone/audio16k'/f'{key}.wav') for key in ids]
    records += [(r.id, r.utterance, Path(r.wav)) for r in rows.itertuples()]
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, (clip_id, text, audio) in enumerate(records):
        samples, rate = sf.read(audio, dtype='float32')
        digest = hashlib.sha256(str(rate).encode()+samples.tobytes()).hexdigest()
        for key in [('id', clip_id), ('text', normalize(text)), ('audio', digest)]:
            if key in seen:
                a, b = find(i), find(seen[key])
                parent[max(a,b)] = min(a,b)
            else:
                seen[key] = i
    groups = np.array([f'g{find(i)}' for i in range(len(records))])
    return groups[:len(ids)], groups[len(ids):]


def metrics(y, p):
    predicted = np.asarray(p) >= .5
    return dict(f1=float(f1_score(y, predicted, zero_division=0)),
                macro_f1=float(f1_score(y, predicted, average='macro', zero_division=0)),
                accuracy=float(accuracy_score(y, predicted)),
                precision=float(precision_score(y, predicted, zero_division=0)),
                recall=float(recall_score(y, predicted, zero_division=0)))


def fit_predict(train, y, groups, test, blocks, config, seed):
    xs, xt = [], []
    for block in blocks:
        sc = StandardScaler().fit(train[block])
        weight = 1/np.sqrt(train[block].shape[1])
        xs.append(np.clip(sc.transform(train[block]), -5, 5)*weight)
        xt.append(np.clip(sc.transform(test[block]), -5, 5)*weight)
    # Keep repeated utterances together even during C selection.
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups))
    for tr, te in splits:
        assert not set(groups[tr]) & set(groups[te])
    classifier = LogisticRegressionCV(Cs=config['lr_Cs'], cv=splits, scoring='neg_log_loss',
                                      class_weight='balanced', max_iter=3000)
    classifier.fit(np.hstack(xs), y)
    return classifier.predict_proba(np.hstack(xt))[:, list(classifier.classes_).index(1)]


def matched_subset(indices, labels, baseline_labels, seed):
    rng = np.random.default_rng(seed)
    keep = []
    for label in (0, 1):
        pool = indices[labels[indices] == label]
        count = int((baseline_labels == label).sum())
        assert len(pool) >= count
        keep.extend(rng.choice(pool, count, replace=False))
    return np.array(sorted(keep))


def run():
    started = time.perf_counter()
    rows = pd.read_csv(DATA / 'metadata.csv')
    baseline = json.loads((ROOT / 'truetone/data/sarcasm_data.json').read_text())
    bundle = joblib.load(ROOT / 'models/baseline/truetone_lr.joblib')
    config = bundle['config']
    ids = np.array(bundle['ids'])
    old_y = np.array([int(baseline[i]['sarcasm']) for i in ids])
    old_groups, new_groups = leakage_groups(rows, baseline, ids)
    new_y = rows.label.to_numpy()
    pd.DataFrame({'id': rows.id, 'group': new_groups}).to_csv(OUT / 'leakage_groups.csv', index=False)
    old = dict(np.load(DATA / 'baseline_features.npz'))
    new = dict(np.load(DATA / 'features.npz'))
    shared = np.where(rows.id.isin(ids))[0]
    shared_y = new_y[shared]
    conditions = ['MUStARD', 'MUStARD++', 'MUStARD++ size-matched']
    predictions, folds = [], []
    for repeat, seed in enumerate([42, 43, 44]):
        outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
        for fold, (_, te_local) in enumerate(outer.split(np.zeros(len(shared)), shared_y, new_groups[shared])):
            te = shared[te_local]
            test_ids = set(rows.id.iloc[te])
            held_groups = set(new_groups[te])
            # Also remove original spellings of held-out shared IDs.
            held_groups.update(old_groups[np.isin(ids, list(test_ids))])
            old_tr = np.where(~np.isin(ids, list(test_ids)) & ~np.isin(old_groups, list(held_groups)))[0]
            new_tr = np.where(~rows.id.isin(test_ids) & ~np.isin(new_groups, list(held_groups)))[0]
            sized_tr = matched_subset(new_tr, new_y, old_y[old_tr], seed*10+fold)
            for condition, arrays, train_idx, labels, groups, train_ids in [
                (conditions[0], old, old_tr, old_y, old_groups, ids),
                (conditions[1], new, new_tr, new_y, new_groups, rows.id.to_numpy()),
                (conditions[2], new, sized_tr, new_y, new_groups, rows.id.to_numpy())]:
                assert not set(train_ids[train_idx]) & test_ids
                assert not set(groups[train_idx]) & held_groups
                for mode, blocks in MODES.items():
                    p = fit_predict({b: arrays[b][train_idx] for b in blocks}, labels[train_idx], groups[train_idx],
                                    {b: new[b][te] for b in blocks}, blocks, config, seed)
                    folds.append(dict(protocol='shared_grouped_cv', repeat=repeat, fold=fold, training=condition,
                                      mode=mode, n_train=len(train_idx), n_test=len(te), **metrics(new_y[te], p)))
                    predictions.extend(dict(protocol='shared_grouped_cv', repeat=repeat, fold=fold, training=condition,
                                            mode=mode, id=rows.id.iloc[i], group=new_groups[i], label=int(new_y[i]), score=float(s))
                                       for i,s in zip(te,p))
                    print('FOLD',repeat,fold,condition,mode,round(folds[-1]['f1'],4),flush=True)
            pd.DataFrame(folds).to_csv(OUT / 'folds.csv', index=False)
            pd.DataFrame(predictions).to_csv(OUT / 'predictions.csv', index=False)
    frame = pd.DataFrame(predictions)
    summary = []
    for (condition, mode), part in frame.groupby(['training','mode']):
        avg = part.groupby('id').agg(label=('label','first'),score=('score','mean'))
        fold_part = pd.DataFrame(folds).query('training == @condition and mode == @mode')
        summary.append(dict(training=condition, mode=mode, n_test=len(avg), **metrics(avg.label, avg.score),
                            mean_fold_f1=float(fold_part.f1.mean()), std_fold_f1=float(fold_part.f1.std())))
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    # Paired bootstrap resamples duplicate-transcript groups, not repeated CV rows.
    intervals = []
    for mode in MODES:
        sub = frame[frame['mode']==mode]
        pivot = sub.groupby(['id','training']).score.mean().unstack()
        labels = sub.groupby('id').label.first().loc[pivot.index].to_numpy()
        groups = sub.groupby('id').group.first().loc[pivot.index].to_numpy()
        unique = np.unique(groups)
        locations = [np.where(groups==g)[0] for g in unique]
        rng = np.random.default_rng(42)
        for candidate in conditions[1:]:
            before = pivot[conditions[0]].to_numpy(); after = pivot[candidate].to_numpy()
            deltas = []
            for _ in range(2000):
                ix = np.concatenate([locations[i] for i in rng.integers(0,len(unique),len(unique))])
                deltas.append(f1_score(labels[ix],after[ix]>=.5,zero_division=0)-f1_score(labels[ix],before[ix]>=.5,zero_division=0))
            intervals.append(dict(mode=mode, comparison=candidate+' minus MUStARD',
                delta_f1=metrics(labels,after)['f1']-metrics(labels,before)['f1'],
                ci_low=float(np.quantile(deltas,.025)),ci_high=float(np.quantile(deltas,.975))))
    pd.DataFrame(intervals).to_csv(OUT / 'paired_bootstrap.csv',index=False)
    # A new-show test addresses generalization beyond random utterance splitting.
    show_results = []
    for show in sorted(rows.show.unique()):
        te = np.where(rows.show.to_numpy()==show)[0]
        excluded = set(new_groups[te])
        excluded.update(old_groups[np.isin(ids,rows.id.iloc[te])])
        for condition, arrays, labels, groups, candidate_ids, shows in [
            ('MUStARD',old,old_y,old_groups,ids,np.array([baseline[i]['show'] for i in ids])),
            ('MUStARD++',new,new_y,new_groups,rows.id.to_numpy(),rows.show.to_numpy())]:
            tr=np.where((shows!=show)&~np.isin(groups,list(excluded))&~np.isin(candidate_ids,rows.id.iloc[te]))[0]
            for mode, blocks in MODES.items():
                p=fit_predict({b:arrays[b][tr] for b in blocks},labels[tr],groups[tr],{b:new[b][te] for b in blocks},blocks,config,42)
                show_results.append(dict(show=show,training=condition,mode=mode,n_train=len(tr),n_test=len(te),
                                         n_positive=int(new_y[te].sum()),**metrics(new_y[te],p)))
                print('SHOW',show,condition,mode,round(show_results[-1]['f1'],4),flush=True)
    pd.DataFrame(show_results).to_csv(OUT / 'held_out_show.csv',index=False)
    from experiment.train import main as train_final
    train_final()
    report=['# MUStARD vs MUStARD++: local paired comparison','',
        'All conditions use identical frozen encoders, LR tuning grid and threshold. No emotion ground-truth annotations are fed as features.',
        'The primary test population is the same 688 shared IDs. Repeated normalized utterances and identical decoded audio are grouped in outer and inner CV.',
        'There are three repetitions of five grouped folds. Added clips may be training examples only when their target group is absent from the test fold.',
        'Size-matched results separate dataset composition from simply having more training examples. Original baseline has its original labels/transcripts.',
        'Shared test features use MUStARD++ transcripts, including six corrected transcripts. Audio for shared IDs is the private original-audio copy.',
        'Block scalers are fitted on the outer training set and reused during inner C selection; inner preprocessing is not fully nested. The outer test set remains excluded.',
        'These are development-set comparisons; confidence intervals do not capture every source of dataset/domain uncertainty. Exact-match grouping cannot eliminate related dialogue, episode or speaker effects. Held-out-show evaluation is reported separately; new speakers and live microphones need a separate labeled benchmark.',
        '', '## Pooled shared-test results (average each clip’s three out-of-fold probabilities)','',pd.DataFrame(summary).to_markdown(index=False),
        '', '## Paired F1 differences (2,000 group bootstrap samples)','',pd.DataFrame(intervals).to_markdown(index=False),
        '', '## Held-out-show evaluation','',pd.DataFrame(show_results).to_markdown(index=False),
        '', 'The app uses the MUStARD++ final refit; MUStARD weights are archived as the baseline. No personal recordings were used in training or evaluation.',
        'A dataset cannot be declared universally better from these scores. Inspect shared-test, size-matched and held-out-show results together.',
        f'Total comparison and final-refit runtime: {(time.perf_counter()-started)/60:.2f} minutes.']
    (OUT/'report.md').write_text('\n\n'.join(report)+'\n')
    print('COMPLETE',OUT/'report.md',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
