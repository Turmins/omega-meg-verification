"""Validate and present frozen results without modifying the experiment."""

from __future__ import annotations

import base64
from html import escape
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from omega_dl.paths import inside_extension, sha256_file
from omega_dl.verification.protocol import fold_dir, load_protocol, read_json, require_test_frozen, save_json
from omega_dl.verification.scoring import choose_threshold, evaluate_scores

METHODS = ["phase", "cnn", "untrained"]
LABELS = {"phase": "Phase", "cnn": "Trained CNN", "untrained": "Untrained CNN"}
COLORS = {"phase": "#167d8d", "cnn": "#b84b36", "untrained": "#777d86"}
KEYS = ["claimed_id", "source_id", "duration_seconds", "start_seconds"]


def read_frame(path):
    return pd.read_csv(path, float_precision="round_trip")


def validate_group(rows, ids, duration, starts, threshold, stage):
    expected = set(itertools.product(ids, ids, [duration], starts))
    actual = list(rows[KEYS].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, duplicate, or unexpected verification pairs")
    if set(rows.stage) != {stage}:
        raise ValueError("Unexpected evaluation stage")
    truth = (rows.claimed_id == rows.source_id).astype(int).to_numpy()
    if not np.array_equal(truth, rows.is_genuine):
        raise ValueError("Incorrect genuine/impostor labels")
    if not np.isfinite(rows.score).all():
        raise ValueError("Non-finite verification score")
    if stage == "test":
        if not np.all(rows.threshold.to_numpy() == threshold):
            raise ValueError("Test threshold differs from development calibration")
        if not np.array_equal(rows.accepted, (rows.score >= threshold).astype(int)):
            raise ValueError("Saved decisions disagree with score >= threshold")


def load_validated(run_root):
    protocol = load_protocol(run_root)
    frames, metrics, training, locks, completion, hashes = [], [], [], [], [], {}
    for fold in range(6):
        path = fold_dir(run_root, fold)
        locks.append(require_test_frozen(run_root, protocol, fold)["frozen_utc"])
        done = read_json(path / "evaluation_complete.json")
        if done["protocol_sha256"] != protocol["protocol_sha256"] or done["fold"] != fold:
            raise ValueError("Test completion manifest mismatch")
        completion.append(done["completed_utc"])
        frame, development, saved_metrics = [read_frame(path / name) for name in ["test_scores.csv", "development_scores.csv", "metrics.csv"]]
        calibration = read_json(path / "thresholds.json")
        if calibration["source_stage"] != "development":
            raise ValueError("Thresholds must originate from development")
        expected_groups = set(itertools.product(METHODS, protocol["config"]["durations"]))
        for table in [frame, development, saved_metrics]:
            if set(table.fold) != {fold} or set(zip(table.method, table.duration_seconds)) != expected_groups:
                raise ValueError("Incomplete methods, durations, or folds")
        for (method, duration), rows in frame.groupby(["method", "duration_seconds"]):
            selected = calibration["thresholds"][f"{method}:{duration}"]
            dev = development[(development.method == method) & (development.duration_seconds == duration)]
            validate_group(dev, protocol["folds"][fold]["val_subjects"], duration, protocol["config"]["query_starts"], selected["threshold"], "development")
            if choose_threshold(dev.is_genuine, dev.score) != selected:
                raise ValueError("Saved threshold is not reproduced by development scores")
            validate_group(rows, protocol["folds"][fold]["test_subjects"], duration, protocol["config"]["query_starts"], selected["threshold"], "test")
            calculated = evaluate_scores(rows.is_genuine, rows.score, selected["threshold"])
            saved = saved_metrics[(saved_metrics.method == method) & (saved_metrics.duration_seconds == duration)]
            if len(saved) != 1 or any(not np.isclose(saved.iloc[0][key], value, rtol=0, atol=1e-14) for key, value in calculated.items()):
                raise ValueError("Saved metrics do not reproduce from decisions")
            metrics.append({"fold": fold, "method": method, "duration_seconds": int(duration), "threshold": selected["threshold"], **calculated})
        frames.append(frame)
        training.append(read_json(path / "training_summary.json"))
        for name in ["test_scores.csv", "development_scores.csv", "metrics.csv", "test_lock.json", "training.csv", "evaluation_complete.json"]:
            hashes[str((path / name).relative_to(run_root))] = sha256_file(path / name)
    if max(locks) > min(completion):
        raise ValueError("A test evaluation finished before all calibrations were frozen")
    test_ids = [s for f in protocol["folds"] for s in f["test_subjects"]]
    if len(test_ids) != 60 or len(set(test_ids)) != 60:
        raise ValueError("Every participant must be tested exactly once")
    return protocol, pd.concat(frames, ignore_index=True), pd.DataFrame(metrics), pd.DataFrame(training), hashes


def summarize(scores, fold_metrics):
    rows = []
    for (method, duration), frame in scores.groupby(["method", "duration_seconds"]):
        genuine, impostor = frame[frame.is_genuine == 1], frame[frame.is_genuine == 0]
        fa, fr = int(impostor.accepted.sum()), int((1 - genuine.accepted).sum())
        far, frr = fa / len(impostor), fr / len(genuine)
        folds = fold_metrics[(fold_metrics.method == method) & (fold_metrics.duration_seconds == duration)]
        rows.append(dict(method=method, duration_seconds=int(duration), n_genuine=len(genuine), n_impostor=len(impostor), false_accepts=fa, false_rejects=fr, FAR=far, FRR=frr, HTER=(far + frr) / 2, TAR=1 - frr, mean_fold_AUC=folds.ROC_AUC.mean(), mean_fold_EER=folds.EER_diagnostic.mean(), min_fold_HTER=folds.HTER.min(), max_fold_HTER=folds.HTER.max()))
    return pd.DataFrame(rows)


def participant_metrics(scores):
    rows = []
    for (method, duration, fold), frame in scores.groupby(["method", "duration_seconds", "fold"]):
        for subject in sorted(frame.claimed_id.unique()):
            own = frame[(frame.claimed_id == subject) & (frame.is_genuine == 1)]
            against = frame[(frame.claimed_id == subject) & (frame.is_genuine == 0)]
            as_source = frame[(frame.source_id == subject) & (frame.is_genuine == 0)]
            rows.append(dict(method=method, duration_seconds=duration, fold=fold, participant_id=subject, genuine_attempts=len(own), false_rejects=int((1 - own.accepted).sum()), FRR=float((1 - own.accepted).mean()), impostor_attempts_against_template=len(against), false_accepts_against_template=int(against.accepted.sum()), FAR_against_template=float(against.accepted.mean()), FAR_as_impostor_source=float(as_source.accepted.mean())))
    return pd.DataFrame(rows)


def permutation_control(scores, protocol, repeats=200):
    rng = np.random.default_rng(81029)
    mappings = [{fold: dict(zip(protocol["folds"][fold]["test_subjects"], rng.permutation(protocol["folds"][fold]["test_subjects"]))) for fold in range(6)} for _ in range(repeats)]
    result = []
    for (method, duration), frame in scores.groupby(["method", "duration_seconds"]):
        grouped = list(frame.groupby("fold"))
        for repetition, by_fold in enumerate(mappings):
            aucs = []
            for fold, rows in grouped:
                labels = (rows.claimed_id.map(by_fold[fold]) == rows.source_id).astype(int)
                aucs.append(roc_auc_score(labels, rows.score))
            result.append(dict(method=method, duration_seconds=int(duration), repetition=repetition, mean_fold_AUC=float(np.mean(aucs))))
    return pd.DataFrame(result)


def figures(run_root, output, scores, metrics, summary, people):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    output.mkdir(parents=True, exist_ok=True)
    names = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=150, bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        names.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ["FAR", "FRR"], ["False acceptance", "False rejection"]):
        for method in METHODS:
            rows = summary[summary.method == method].sort_values("duration_seconds")
            axis.plot(rows.duration_seconds, rows[metric] * 100, "o-", color=COLORS[method], label=LABELS[method])
        axis.set(title=title, xlabel="Query duration (seconds)", ylabel="Attempts (%)", xticks=[10, 30, 60], ylim=(0, 100))
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Test errors at frozen development thresholds", y=1.03)
    save(fig, "duration_errors")

    fig, ax = plt.subplots(figsize=(10, 4))
    for method in METHODS:
        rows = metrics[(metrics.method == method) & (metrics.duration_seconds == 30)].sort_values("fold")
        ax.plot(rows.fold, rows.HTER * 100, "o-", label=LABELS[method], color=COLORS[method])
    ax.set(title="30-second HTER by held-out fold", xlabel="Fold", ylabel="HTER (%)", xticks=range(6), ylim=(0, 100))
    ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    save(fig, "fold_errors")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for fold in range(6):
        history = read_frame(fold_dir(run_root, fold) / "training.csv")
        axes[0].plot(history.epoch, history.loss, label=f"Fold {fold}")
        valid = history.dropna(subset=["development_AUC_30s"])
        axes[1].plot(valid.epoch, valid.development_AUC_30s, label=f"Fold {fold}")
    axes[0].set(title="Training objective", xlabel="Epoch", ylabel="Triplet + auxiliary loss")
    axes[1].set(title="Development selection metric", xlabel="Epoch", ylabel="30-second ROC-AUC", ylim=(0, 1))
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    save(fig, "training")

    fig, axes = plt.subplots(6, 2, figsize=(11, 14))
    for fold in range(6):
        for column, method in enumerate(["phase", "cnn"]):
            rows = scores[(scores.fold == fold) & (scores.method == method) & (scores.duration_seconds == 30)]
            ax = axes[fold, column]
            bins = np.linspace(rows.score.min(), rows.score.max(), 24)
            for label, color, name in [(0, "#999faa", "Impostor"), (1, "#167d8d", "Genuine")]:
                ax.hist(rows.loc[rows.is_genuine == label, "score"], bins=bins, density=True, alpha=.65, color=color, label=name)
            ax.axvline(rows.threshold.iloc[0], color="#b84b36", linestyle="--", label="Dev threshold")
            ax.set(title=f"{LABELS[method]} | fold {fold}", xlabel="Similarity", ylabel="Density")
            if fold == 0:
                ax.legend(fontsize=7, frameon=False)
    fig.suptitle("30-second score distributions: separate scales for each fold", y=1.01)
    save(fig, "score_distributions")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for method in ["phase", "cnn"]:
        rows = people[(people.method == method) & (people.duration_seconds == 30)].sort_values("participant_id")
        for ax, metric in zip(axes, ["FRR", "FAR_against_template"]):
            ax.plot(range(1, 61), rows[metric] * 100, ".", label=LABELS[method], color=COLORS[method], alpha=.8)
    axes[0].set(title="Participant variation at 30 seconds", ylabel="False rejection (%)", ylim=(-3, 103))
    axes[1].set(xlabel="Participant index (sorted ID; see participant_metrics.csv)", ylabel="False acceptance\nagainst template (%)", ylim=(-3, 103))
    axes[0].legend(frameon=False)
    save(fig, "participant_errors")
    return names


CSS = """body{margin:0;background:#f1f4f6;color:#172735;font:16px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:40px 28px 80px}h1{font-size:38px;line-height:1.15;max-width:900px}h2{margin-top:36px;font-size:25px}h3{font-size:19px}.eyebrow{color:#167d8d;letter-spacing:.15em;font-size:12px;font-weight:800}.lead{font-size:20px;max-width:900px}.note{color:#53616e}.card,figure{background:white;border:1px solid #dbe2e6;border-radius:12px;padding:22px;margin:18px 0}figure img{width:100%;height:auto}figcaption{color:#53616e;font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #dbe2e6}th{background:#eaf1f3}a{color:#126778}code{font-size:13px;overflow-wrap:anywhere}.scroll{overflow-x:auto}.flow{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.flow span{padding:16px;background:#e9f2f3;border-radius:8px;flex:1;min-width:140px}.controls{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:18px}label{font-size:14px;font-weight:600}select,button{font:inherit;padding:10px 14px;border-radius:7px;border:1px solid #abb9c1}select{display:block;width:100%;margin-top:6px;background:white}button{background:#155d6b;color:white;cursor:pointer;margin:18px 10px 0 0}button.secondary{background:white;color:#155d6b}.results{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:18px}.decision{font-size:27px;font-weight:750}.accept{color:#15714d}.reject{color:#ab422e}footer{font-size:13px;color:#53616e;margin-top:30px}@media(max-width:650px){main{padding:25px 16px}h1{font-size:30px}}"""


def demo_html(output, protocol, scores):
    data = {}
    for row in scores[scores.method.isin(["phase", "cnn"])].itertuples():
        key = f"{row.fold}|{row.claimed_id}|{row.source_id}|{row.duration_seconds}|{row.start_seconds}"
        data.setdefault(key, {})[row.method] = [row.score, row.threshold, row.accepted]
    payload = json.dumps({"folds": [f["test_subjects"] for f in protocol["folds"]], "pairs": data}, separators=(",", ":")).replace("<", "\\u003c")
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MEG verification demo</title><style>__CSS__</style><main>
<div class="eyebrow">EDUCATIONAL EXPERIMENT / SAVED TEST REPLAY</div><h1>Does this query match the claimed person?</h1>
<p class="lead">One identity claim. One query. Two independently calibrated methods.</p>
<p class="note">This offline page replays all saved held-out decisions. Enrollment uses 240 seconds from another session. No learning or threshold selection occurs here. For inference from the original signals, use the single-claim command in the guide.</p>
<div class="card controls"><label>Held-out fold<select id="fold"></select></label><label>Claimed participant<select id="claimed"></select></label><label>Query source<select id="source"></select></label><label>Query duration<select id="duration"><option>10</option><option selected>30</option><option>60</option></select></label><label>Start within central interval<select id="start"><option>0</option><option>60</option><option>120</option><option>180</option></select></label></div>
<button id="verify">Verify claim</button><button class="secondary" id="own">Genuine example</button><button class="secondary" id="other">Impostor example</button><button class="secondary" id="error">Find an error</button>
<p id="truth" aria-live="polite"></p><div class="results" id="results" aria-live="polite"></div>
<p class="note">Rule: accept when score is at least the development threshold. Scores are similarity values, not probabilities. Display rounding never changes the saved decision. Ground truth is shown only for explanation.</p>
<footer><a href="report.html">Open the full report</a> · All selectable people belong to the test group of the selected model. The error button cycles through errors in both methods, starting at the primary 30-second duration.</footer></main>
<script id="experiment-data" type="application/json">__DATA__</script><script>
'use strict';const D=JSON.parse(document.getElementById('experiment-data').textContent);const el=id=>document.getElementById(id);let errorCursor=0;
function options(id,values){el(id).replaceChildren(...values.map(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;return o;}));}
options('fold',D.folds.map((_,i)=>i));
function populate(){options('claimed',D.folds[+el('fold').value]);options('source',D.folds[+el('fold').value]);el('truth').textContent='Selection changed. Choose Verify claim.';el('results').replaceChildren();}
function key(){return ['fold','claimed','source','duration','start'].map(id=>el(id).value).join('|');}
function verify(){const pair=D.pairs[key()];if(!pair)throw Error('Missing saved pair');const genuine=el('claimed').value===el('source').value;el('truth').textContent=(genuine?'Genuine attempt':'Impostor attempt')+' — source '+el('source').value+' claims '+el('claimed').value+'; '+el('duration').value+' seconds from offset '+el('start').value+'.';el('results').replaceChildren();for(const [method,title] of [['phase','Phase method'],['cnn','Trained CNN']]){const [score,threshold,accepted]=pair[method];const card=document.createElement('article');card.className='card';const h=document.createElement('h2');h.textContent=title;const result=document.createElement('p');result.className='decision '+(accepted?'accept':'reject');result.textContent=accepted?'ACCEPT':'REJECT';const numbers=document.createElement('p');numbers.textContent='Score '+score.toFixed(6)+' / threshold '+threshold.toFixed(6);const accuracy=document.createElement('p');accuracy.textContent=Boolean(accepted)===genuine?'Correct for this attempt':(accepted?'False acceptance':'False rejection');card.append(h,result,numbers,accuracy);el('results').append(card);}}
el('fold').addEventListener('change',populate);for(const id of ['claimed','source','duration','start'])el(id).addEventListener('change',()=>{el('truth').textContent='Selection changed. Choose Verify claim.';el('results').replaceChildren();});el('verify').onclick=verify;
el('own').onclick=()=>{el('source').value=el('claimed').value;verify();};el('other').onclick=()=>{el('source').value=D.folds[+el('fold').value].find(id=>id!==el('claimed').value);verify();};
const errors=Object.keys(D.pairs).filter(k=>{const p=k.split('|'),truth=p[1]===p[2];return Object.values(D.pairs[k]).some(v=>Boolean(v[2])!==truth);}).sort((a,b)=>(a.split('|')[3]==='30'?-1:0)-(b.split('|')[3]==='30'?-1:0)||a.localeCompare(b));
el('error').onclick=()=>{if(!errors.length){el('truth').textContent='No errors in the saved test decisions.';return;}const p=errors[errorCursor++%errors.length].split('|');el('fold').value=p[0];populate();['claimed','source','duration','start'].forEach((id,i)=>el(id).value=p[i+1]);verify();};populate();verify();
</script></html>"""
    (output / "demo.html").write_text(page.replace("__CSS__", CSS).replace("__DATA__", payload), encoding="utf-8")


def render_report(output, protocol, summary, metrics, training, permutations, names):
    primary = summary[summary.duration_seconds == 30].set_index("method")
    delta = (primary.loc["cnn", "HTER"] - primary.loc["phase", "HTER"]) * 100
    learning = (primary.loc["cnn", "HTER"] - primary.loc["untrained", "HTER"]) * 100
    conclusion = f"At 30 seconds, trained CNN HTER is {primary.loc['cnn', 'HTER']:.1%}, versus {primary.loc['phase', 'HTER']:.1%} for phase verification ({abs(delta):.1f} percentage points {'higher' if delta > 0 else 'lower'}). Compared with the untrained encoder, CNN HTER is {abs(learning):.1f} percentage points {'higher' if learning > 0 else 'lower'}."
    duration_text = " ".join(f"{LABELS[method]} HTER changes from {summary[(summary.method == method) & (summary.duration_seconds == 10)].iloc[0].HTER:.1%} at 10 seconds to {summary[(summary.method == method) & (summary.duration_seconds == 60)].iloc[0].HTER:.1%} at 60 seconds." for method in ["phase", "cnn"])
    duration_text += " Each duration has its own development threshold, so this describes the complete verification procedure, including calibration."
    variation_text = " ".join(f"At 30 seconds, {LABELS[method]} fold HTER ranges from {primary.loc[method, 'min_fold_HTER']:.1%} to {primary.loc[method, 'max_fold_HTER']:.1%}." for method in ["phase", "cnn"])
    variation_text += " Participant plots use only four genuine attempts per person; a single rejection changes individual FRR by 25 percentage points. These ranges are descriptive, not confidence intervals."
    table = summary.copy()
    table["method"] = table.method.map(LABELS)
    table = table[["method", "duration_seconds", "false_accepts", "false_rejects", "FAR", "FRR", "HTER", "mean_fold_AUC", "mean_fold_EER"]]
    for col in ["FAR", "FRR", "HTER", "mean_fold_EER"]:
        table[col] = table[col].map(lambda x: f"{x:.2%}")
    table["mean_fold_AUC"] = table.mean_fold_AUC.map(lambda x: f"{x:.4f}")
    table.columns = ["Method", "Query (s)", "False accepts / 2160", "False rejects / 240", "FAR", "FRR", "HTER", "Mean fold AUC", "Mean fold EER"]
    train_table = training[["fold", "seed", "epochs_completed", "steps", "best_epoch", "best_development_AUC_30s", "parameter_count", "elapsed_seconds", "peak_cuda_bytes"]].copy()
    train_table["best_development_AUC_30s"] = train_table.best_development_AUC_30s.round(4)
    train_table["elapsed_seconds"] = train_table.elapsed_seconds.round(1)
    control = permutations.groupby(["method", "duration_seconds"]).mean_fold_AUC.agg(["mean", "std", "min", "max"]).reset_index().round(4)
    control["method"] = control.method.map(LABELS)
    md_table = lambda frame: "| " + " | ".join(map(str, frame.columns)) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False, name=None))
    sections = [
        ("Research question", "Can a short MEG query verify a claimed participant, and does a learned embedding improve on a mathematical phase template? A shared CNN is trained on other participants; enrollment averages 240 seconds from the claimed person's enrollment recording. Query duration is 10, 30, or 60 seconds from the other session. No per-user neural network is fitted."),
        ("Data and evaluation", "Only the supplied 60 participants and 120 recordings are used: 260 channels at 128 Hz. Six folds contain 40 training, 10 development, and 10 test identities, with both sessions kept together. Query starts are 0, 60, 120, and 180 seconds within the central interval. Each method and duration has 240 genuine and 2,160 impostor comparisons across folds. These are dependent comparisons among 60 people."),
        ("Methods", "Both branches independently filter 10-second blocks into theta, alpha, and beta. Phase verification averages band coherencies over permitted blocks and compares phases using mean circular cosine. The CNN uses per-channel temporal median/MAD normalization before float16 caching, 16 learned spatial filters per band, four temporal blocks, statistics pooling, and 128-dimensional unit embeddings. Enrollment and query embeddings are averaged and renormalized before cosine comparison. This phase adaptation differs from the historical 60-second identification procedure."),
        ("Calibration and interpretation", "The checkpoint maximizes development AUC at 30 seconds; earliest ties win. Each method/duration/fold threshold minimizes development |FAR-FRR|, then HTER, then prefers the higher threshold. All six checkpoints and calibrations were frozen before test scoring. Test FAR/FRR use these saved thresholds. FAR is the fraction of impostor attempts accepted; FRR is the fraction of genuine attempts rejected; HTER is their arithmetic mean. AUC and approximate EER are averaged within folds, without combining model score scales. Test EER is diagnostic, not a deployment threshold."),
        ("Limits", "This is exploratory cross-validation on one previously studied dataset, with two sessions per participant and one base seed. Attempts and folds are dependent. No confidence interval based on independent rows is claimed. This experiment does not establish reliability on unavailable sessions, devices, populations, or adversarial attacks. The result does not establish that a different model must perform similarly; no further model selection was made using these test results."),
        ("Reproducibility", "See ../../../docs/VERIFICATION_GUIDE.md, ../protocol.json, ../environment.json, and validation.json. The report independently recomputes thresholds, decisions, pair coverage, and metrics from saved tables and verifies checkpoint locks. All source data and main-project files remain outside the writable experiment area. The isolation audit is stored in ../isolation/."),
    ]
    captions = {"duration_errors": "Errors at each fold's development threshold; enrollment stays at 240 seconds.", "fold_errors": "Each fold has ten test participants. Variation across folds is descriptive.", "training": "Development AUC chooses checkpoints; test outcomes do not affect training.", "score_distributions": "Genuine and impostor densities are normalized separately. Every panel has its own score scale; dashed lines are development thresholds.", "participant_errors": "Each participant has four genuine attempts and 36 impostor attempts against their template. The same ordering is used for both methods."}
    md = ["# Educational MEG verification: experimental report", conclusion, "## Observed test results", md_table(table)]
    content = ["<div class='eyebrow'>MEG BIOMETRICS / SIX-FOLD EXPLORATORY STUDY</div><h1>Verification from short MEG queries</h1>", f"<p class='lead'>{escape(conclusion)}</p>", "<p><a href='demo.html'>Open interactive demonstration</a> · <a href='summary.csv'>Download summary data</a></p>"]
    for title, paragraph in sections[:4]:
        md += [f"## {title}", paragraph]
        content += [f"<h2>{title}</h2><p>{escape(paragraph)}</p>"]
        if title == "Research question":
            content += ["<div class='flow'><span>Train common encoder<br><b>40 people</b></span>→<span>Select model and thresholds<br><b>10 development people</b></span>→<span>Enroll and verify<br><b>10 held-out people</b></span></div>"]
        if title == "Methods":
            content += ["<div class='flow'><span>3 × 260 × 1280<br>band/channel/time</span>→<span>16 spatial filters/band<br>4 temporal blocks</span>→<span>Mean + SD pooling<br>128-D unit vector</span>→<span>Template similarity<br>Accept / reject</span></div>"]
    content += ["<h2>Observed test results</h2><div class='card scroll'>" + table.to_html(index=False, border=0) + "</div>", f"<p>{escape(duration_text)}</p><p>{escape(variation_text)}</p>"]
    md += [duration_text, variation_text]
    for name in names:
        encoded = base64.b64encode((output / "figures" / f"{name}.png").read_bytes()).decode()
        content += [f"<figure><img alt='{escape(captions[name])}' src='data:image/png;base64,{encoded}'><figcaption>{escape(captions[name])}</figcaption></figure>"]
        md += [f"![{captions[name]}](figures/{name}.png)"]
    control_text = "200 random permutations per method/duration change template-to-query identity correspondence while retaining the scores and fold structure. The table describes the distribution of mean fold AUC. It is a diagnostic calculation, not a significance test; no single permutation is required to equal 0.5. The untrained encoder is separately calibrated on development data."
    content += ["<h2>Controls</h2><p>" + control_text + "</p><div class='card scroll'>" + control.to_html(index=False, border=0) + "</div>", "<h2>Training record</h2><div class='card scroll'>" + train_table.to_html(index=False, border=0) + "</div>"]
    md += ["## Controls", control_text, md_table(control), "## Training record", md_table(train_table)]
    for title, paragraph in sections[4:]:
        content += [f"<h2>{title}</h2><p>{escape(paragraph)}</p>"]
        md += [f"## {title}", paragraph]
    content += [f"<footer>Protocol SHA-256: <code>{protocol['protocol_sha256']}</code><br>Source tables: summary.csv, fold_metrics.csv, participant_metrics.csv, permutation_control.csv. No scores were altered for presentation.</footer>"]
    (output / "report.html").write_text("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>MEG verification report</title><style>" + CSS + "</style><main>" + "\n".join(content) + "</main></html>", encoding="utf-8")
    (output / "report.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    return conclusion


def build_report(run_root):
    run_root = inside_extension(run_root)
    output = inside_extension(run_root / "report")
    output.mkdir(parents=True, exist_ok=True)
    protocol, scores, metrics, training, hashes = load_validated(run_root)
    summary, people = summarize(scores, metrics), participant_metrics(scores)
    permutations = permutation_control(scores, protocol)
    for name, frame in [("summary", summary), ("fold_metrics", metrics), ("participant_metrics", people), ("permutation_control", permutations), ("training_summary", training)]:
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.17g")
    names = figures(run_root, output / "figures", scores, metrics, summary, people)
    demo_html(output, protocol, scores)
    primary_pairs = scores[(scores.duration_seconds == 30) & scores.method.isin(["phase", "cnn"])].sort_values(["fold", "claimed_id", "source_id", "start_seconds", "method"])
    example_keys = []
    for selected in [primary_pairs[primary_pairs.is_genuine == 1], primary_pairs[primary_pairs.is_genuine == 0], primary_pairs[primary_pairs.accepted != primary_pairs.is_genuine]]:
        if len(selected):
            row = selected.iloc[0]
            key = (int(row.fold), row.claimed_id, row.source_id, int(row.start_seconds))
            if key not in example_keys:
                example_keys.append(key)
    examples = []
    for fold, claimed, source, start in example_keys:
        rows = primary_pairs[(primary_pairs.fold == fold) & (primary_pairs.claimed_id == claimed) & (primary_pairs.source_id == source) & (primary_pairs.start_seconds == start)]
        examples.extend(rows.to_dict(orient="records"))
    save_json(output / "presentation_data.json", {"protocol_sha256": protocol["protocol_sha256"], "summary": summary.to_dict(orient="records"), "fold_metrics": metrics.to_dict(orient="records"), "training": training.to_dict(orient="records"), "examples": examples, "example_selection": "First sorted genuine, impostor, and erroneous comparison at 30 seconds, deduplicated"})
    conclusion = render_report(output, protocol, summary, metrics, training, permutations, names)
    save_json(output / "validation.json", {"status": "passed", "protocol_sha256": protocol["protocol_sha256"], "source_hashes": hashes, "reporting_code_sha256": sha256_file(Path(__file__)), "folds": 6, "participants": 60, "score_rows": len(scores), "metric_rows": len(metrics), "pairs_per_method_duration": 2400, "checks": ["all frozen checkpoint/calibration hashes", "development threshold recomputation", "complete and identical comparison grids", "genuine/impostor labels", "test threshold provenance", "full-precision decision recomputation", "saved metric recomputation", "all calibrations precede first test completion", "each test participant appears once"], "conclusion": conclusion})
    return conclusion
