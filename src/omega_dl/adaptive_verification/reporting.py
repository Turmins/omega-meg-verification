"""Generate inspectable metrics and a compact replay of sequential verification."""
from __future__ import annotations
import html
import json
import numpy as np
import pandas as pd
from omega_dl.paths import extension_root
from omega_dl.verification.protocol import read_json, save_json
from omega_dl.verification.scoring import choose_threshold, evaluate_scores
from .protocol import fold_path, require_frozen
from .experiment import records_for
from .evaluation import stream_scores, summarize_streams, POLICIES
from .online import strict_update_threshold


def read_csv(path):
    return pd.read_csv(path, float_precision="round_trip")


def aggregate(frame, stream=False):
    keys = ["method", "enrollment_seconds", "duration_seconds"] + (["policy"] if stream else [])
    out = []
    for group, rows in frame.groupby(keys):
        row = dict(zip(keys, group))
        ng, ni = int(rows.n_genuine.sum()), int(rows.n_impostor.sum())
        fa, fr = int(rows.false_accepts.sum()), int(rows.false_rejects.sum())
        row.update(n_genuine=ng, n_impostor=ni, false_accepts=fa, false_rejects=fr,
                   FAR=fa / ni, FRR=fr / ng, HTER=(fa / ni + fr / ng) / 2)
        if stream:
            row.update(updates=int(rows.updates.sum()), impostor_updates=int(rows.impostor_updates.sum()),
                       mean_final_drift=float(rows.mean_final_drift.mean()))
        else:
            row.update(mean_fold_AUC=float(rows.ROC_AUC.mean()),
                       mean_fold_EER=float(rows.EER_diagnostic.mean()))
        out.append(row)
    return pd.DataFrame(out)


def validate(root, p, base):
    lock = require_frozen(root, p)
    metrics, streams, training = [], [], []
    for fold in range(6):
        path = fold_path(root, fold)
        completed = read_json(path / "evaluation_complete.json")
        if completed["protocol_sha256"] != p["protocol_sha256"] or completed["completed_utc"] < lock["frozen_utc"]:
            raise ValueError("Test evaluation was not after the global freeze")
        scores = read_csv(path / "static_scores.csv")
        actual_metrics = read_csv(path / "static_metrics.csv")
        if not np.isfinite(scores.score).all() or not np.array_equal(
                scores.is_genuine.to_numpy(), (scores.claimed_id == scores.source_id).astype(int).to_numpy()):
            raise ValueError("Invalid static labels or scores")
        keys = ["method", "enrollment_seconds", "duration_seconds", "claimed_id", "source_id", "start_seconds"]
        if scores.duplicated(keys).any():
            raise ValueError("Duplicate static attempts")
        for (method, enrollment, duration), group in scores.groupby(keys[:3]):
            if len(group) != 100 * (240 // duration) or set(group.start_seconds) != set(range(0, 240, duration)):
                raise ValueError("Incomplete static comparison grid")
            if method == "phase":
                file = path / "phase_thresholds.json"
            else:
                original = method.removesuffix("_initial")
                file = fold_path(root, fold, original) / ("initial_thresholds.json" if method.endswith("_initial") else "thresholds.json")
            entry = read_json(file)["thresholds"][f"{enrollment}:{duration}"]
            if not np.all(group.threshold == entry["threshold"]) or not np.array_equal(group.accepted, group.score >= group.threshold):
                raise ValueError("Static decisions or thresholds differ")
            computed = evaluate_scores(group.is_genuine, group.score, entry["threshold"])
            saved = actual_metrics[(actual_metrics.method == method) & (actual_metrics.enrollment_seconds == enrollment)
                                   & (actual_metrics.duration_seconds == duration)]
            if len(saved) != 1 or any(not np.isclose(float(saved.iloc[0][k]), v, atol=1e-12, rtol=0) for k, v in computed.items()):
                raise ValueError("Static metric mismatch")
        records, ids = records_for(base, p, fold, "test", root)
        saved_stream = read_csv(path / "stream_scores.csv")
        replay = []
        for method in p["config"]["models"]:
            method_path = fold_path(root, fold, method)
            dev = read_csv(method_path / "development_scores.csv")
            thresholds = read_json(method_path / "thresholds.json")["thresholds"]
            for (enrollment, duration), group in dev.groupby(["enrollment_seconds", "duration_seconds"]):
                computed = choose_threshold(group.is_genuine, group.score)["threshold"]
                entry = thresholds[f"{enrollment}:{duration}"]
                if computed != entry["threshold"]:
                    raise ValueError("Threshold was not reproduced from development data")
                strict = strict_update_threshold(group.is_genuine, group.score, computed,
                                                  p["config"]["adaptation"]["update_development_far"])
                gate_far = float((group.loc[group.is_genuine == 0, "score"] >= strict).mean())
                if strict != entry["update_threshold"] or gate_far != entry["empirical_update_gate_FAR"]:
                    raise ValueError("Update gate was not reproduced from development data")
            with np.load(method_path / "test_embeddings.npz", allow_pickle=False) as values:
                features = {key: values[key] for key in values.files}
            replay.append(stream_scores(features, records, ids, p["config"], thresholds, fold, method))
            training.append(read_json(method_path / "training_summary.json"))
        replay = pd.concat(replay, ignore_index=True)
        if list(replay.columns) != list(saved_stream.columns) or len(replay) != len(saved_stream):
            raise ValueError("Stream schema changed")
        for column in replay.columns:
            a, b = replay[column], saved_stream[column]
            if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
                if not np.allclose(a, b, atol=1e-12, rtol=0):
                    raise ValueError(f"Sequential replay differs: {column}")
            elif not np.array_equal(a, b):
                raise ValueError(f"Sequential replay differs: {column}")
        recomputed = summarize_streams(replay).sort_values(["method", "enrollment_seconds", "duration_seconds", "policy"]).reset_index(drop=True)
        saved = read_csv(path / "stream_metrics.csv").sort_values(["method", "enrollment_seconds", "duration_seconds", "policy"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(recomputed, saved, check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
        metrics.append(actual_metrics)
        streams.append(saved_stream)
    return pd.concat(metrics, ignore_index=True), pd.concat(streams, ignore_index=True), pd.DataFrame(training)


STYLE = """body{font:16px/1.55 system-ui,sans-serif;color:#172c3a;background:#f3f6f8;margin:0}main{max-width:1200px;margin:auto;padding:32px}h1{font-size:32px;line-height:1.2}h2{margin-top:32px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:9px 12px;text-align:right;border-bottom:1px solid #dde5ea}th:first-child,td:first-child{text-align:left}th{background:#123947;color:white}a{color:#006b70}section,.note{background:white;padding:20px;border-radius:12px;margin:18px 0}.controls{display:flex;gap:12px;flex-wrap:wrap}select{padding:7px;font:inherit}small{color:#496575}.accept{color:#0a7559}.reject{color:#ae3f3f}.scroll{overflow:auto}"""


def table(frame, columns):
    shown = frame[columns].copy()
    for name in ("FAR", "FRR", "HTER", "mean_fold_EER"):
        if name in shown:
            shown[name] = shown[name].map(lambda v: f"{100 * v:.2f}%")
    if "mean_fold_AUC" in shown:
        shown["mean_fold_AUC"] = shown.mean_fold_AUC.map(lambda v: f"{v:.4f}")
    return shown.to_html(index=False, border=0, escape=True)


def build_demo(root, streams):
    subset = streams[streams.repetition == 0]
    subjects = sorted(set(subset.claimed_id) | set(subset.source_id))
    methods = ["eegnet", "graph"]
    payload = []
    for row in subset.itertuples(index=False):
        payload.append([methods.index(row.method), int(row.fold), int(row.enrollment_seconds), int(row.duration_seconds),
                        POLICIES.index(row.policy), subjects.index(row.claimed_id), subjects.index(row.source_id),
                        int(row.start_seconds), float(row.score), float(row.threshold), float(row.update_threshold),
                        int(row.accepted), int(row.updated), float(row.drift_from_anchor)])
    data = json.dumps({"subjects": subjects, "rows": payload}, separators=(",", ":"))
    page = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Adaptive verification replay</title><style>__STYLE__</style><main><a href="report.html">Back to results</a><h1>Watch a personal reference evolve</h1><p>Recorded test replay: repetition 0, with genuine and impostor queries in chronological order. Decisions are made before updates. This page does not run live model inference.</p><section class="controls"><label>Encoder <select id="model"><option value="0">EEGNet</option><option value="1">Graph</option></select></label><label>Fold <select id="fold"></select></label><label>Enrollment <select id="enroll"><option>30</option><option>240</option></select> s</label><label>Query <select id="duration"><option>10</option><option selected>30</option><option>60</option></select> s</label><label>Policy <select id="policy"><option value="0">Static</option><option value="1">Update accepted</option><option value="2" selected>Guarded update</option></select></label><label>Claimed person <select id="claim"></select></label></section><p id="summary"></p><div class="scroll"><table><thead><tr><th>Start</th><th>Source</th><th>Truth</th><th>Score</th><th>Accept threshold</th><th>Update threshold</th><th>Decision</th><th>Updated</th><th>Reference drift</th></tr></thead><tbody id="body"></tbody></table></div><p><small>Scores are similarities, not probabilities. Reference drift is 1 minus cosine similarity to the original enrollment template. The guarded rule also checks similarity to the original template. Five stream repetitions are included in the CSV results; this replay shows one.</small></p></main><script id="data" type="application/json">__DATA__</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const el=id=>document.getElementById(id), value=id=>Number(el(id).value);
for(let f=0;f<6;f++)el('fold').add(new Option(f,f));
function claims(){const previous=el('claim').value;el('claim').replaceChildren();[...new Set(data.rows.filter(r=>r[1]===value('fold')).map(r=>r[5]))].sort((a,b)=>a-b).forEach(s=>el('claim').add(new Option(data.subjects[s],s)));if([...el('claim').options].some(o=>o.value===previous))el('claim').value=previous;}
function render(){const rows=data.rows.filter(r=>r[0]===value('model')&&r[1]===value('fold')&&r[2]===value('enroll')&&r[3]===value('duration')&&r[4]===value('policy')&&r[5]===value('claim'));el('body').replaceChildren();let errors=0,updates=0,wrongUpdates=0;for(const r of rows){const genuine=r[5]===r[6];errors+=Number(Boolean(r[11])!==genuine);updates+=r[12];wrongUpdates+=Number(r[12]&&!genuine);const tr=document.createElement('tr');[r[7]+' s',data.subjects[r[6]],genuine?'Genuine':'Impostor',r[8].toFixed(4),r[9].toFixed(4),r[10].toFixed(4),r[11]?'YES':'NO',r[12]?'Yes':'No',r[13].toFixed(6)].forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===6)td.className=r[11]?'accept':'reject';tr.append(td);});el('body').append(tr);}el('summary').textContent=rows.length+' attempts · '+errors+' errors · '+updates+' updates · '+wrongUpdates+' updates from impostor attempts';}
for(const id of ['model','fold','enroll','duration','policy','claim'])el(id).addEventListener('change',()=>{if(id==='fold')claims();render();});
claims();render();
</script></html>"""
    (root / "report/demo.html").write_text(page.replace("__STYLE__", STYLE).replace("__DATA__", data.replace("</", "<\\/")), encoding="utf-8")


def observed_results(static, sequential):
    def row(frame, method, policy=None):
        selected = frame[(frame.method == method) & (frame.enrollment_seconds == 30)
                         & (frame.duration_seconds == 30)]
        if policy is not None:
            selected = selected[selected.policy == policy]
        return selected.iloc[0]
    cnn, graph, phase = (row(static, method) for method in ("eegnet", "graph", "phase"))
    paragraphs = [
        f"With 30-second enrollment and 30-second queries, static HTER is {cnn.HTER:.2%} for EEGNet, "
        f"{graph.HTER:.2%} for the graph model, and {phase.HTER:.2%} for the phase control. "
        "EEGNet is the stronger neural encoder in this run; neither trained encoder beats the phase control at this setting.",
        f"Training lowers HTER from {row(static, 'eegnet_initial').HTER:.2%} to {cnn.HTER:.2%} for EEGNet "
        f"and from {row(static, 'graph_initial').HTER:.2%} to {graph.HTER:.2%} for the graph model. "
        "Both learn useful information relative to their separately calibrated initial-weight controls."
    ]
    for method, label in (("eegnet", "EEGNet"), ("graph", "Graph")):
        fixed, accepted, guarded = (row(sequential, method, policy) for policy in ("static", "accepted", "guarded"))
        paragraphs.append(
            f"{label}, matched 30/30-second streams: HTER is {fixed.HTER:.2%} without updates, "
            f"{accepted.HTER:.2%} when every accepted query updates the template, and {guarded.HTER:.2%} "
            f"with the guarded rule. Guarding reduces impostor-origin updates from {int(accepted.impostor_updates)} "
            f"to {int(guarded.impostor_updates)} across 1,200 impostor attempts. It does not eliminate them.")
    paragraphs.append(
        "The guarded rule barely changes HTER at this setting. Its measured benefit is fewer contaminated updates, "
        "not a demonstrated accuracy gain. The 1% development update-gate target is not a guarantee on held-out people. "
        "Stream percentages use a different sampled set of attempts from static all-pairs percentages.")
    return paragraphs



def build_report(root, p):
    from .protocol import load
    _, base = load(root)
    metrics, streams, training = validate(root, p, base)
    static = aggregate(metrics)
    stream_metrics = summarize_streams(streams)
    sequential = aggregate(stream_metrics, stream=True)
    folder = root / "report"
    folder.mkdir(parents=True, exist_ok=True)
    static.to_csv(folder / "static_summary.csv", index=False, float_format="%.17g")
    sequential.to_csv(folder / "adaptive_summary.csv", index=False, float_format="%.17g")
    metrics.to_csv(folder / "fold_static_metrics.csv", index=False, float_format="%.17g")
    stream_metrics.to_csv(folder / "fold_adaptive_metrics.csv", index=False, float_format="%.17g")
    training.to_csv(folder / "training_summary.csv", index=False)
    primary = static[static.duration_seconds == 30]
    adaptive = sequential[sequential.duration_seconds == 30]
    columns = ["method", "enrollment_seconds", "FAR", "FRR", "HTER", "mean_fold_AUC"]
    adaptive_columns = ["method", "enrollment_seconds", "policy", "FAR", "FRR", "HTER", "updates", "impostor_updates"]
    introduction = "Two compact encoders share the same cross-session prototype loss, identity splits, and template rules. Only the existing 60 participants and two recordings per person are used. All twelve checkpoints and development thresholds were frozen before any new test evaluation."
    limitations = "This is exploratory reuse of a previously studied dataset, with one training seed per fold. V2 uses a different query grid from v1; the phase control is recalibrated here for a matched comparison. Windows and repeated streams are dependent observations. The graphs describe sensor-signal relationships, not anatomical connections. Adaptation is simulated within the second session; long-term performance cannot be established from two sessions. Scores are not calibrated probabilities."
    explanation = "Static comparisons use every held-out claimed/source pair and nonoverlapping query windows. Sequential results use five balanced genuine/impostor schedules per claim, with separate template state for every model, duration, enrollment length, repetition and policy. Each query is scored before any update. Ground truth is never supplied to the update rule."
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>EEGNet and graph verification</title><style>' + STYLE + '</style><main><h1>EEGNet and graph verification</h1><p>' + introduction + '</p><p><a href="demo.html">Explore the sequential replay</a> · <a href="static_summary.csv">All static results</a> · <a href="adaptive_summary.csv">All adaptive results</a></p><section><h2>Static verification · 30-second queries</h2><div class="scroll">' + table(primary, columns) + '</div></section><section><h2>Sequential verification · 30-second queries</h2><p>' + explanation + '</p><div class="scroll">' + table(adaptive, adaptive_columns) + '</div></section><section><h2>Training</h2>' + table(training, ["model", "fold", "parameter_count", "best_epoch", "epochs", "elapsed_seconds"]) + '</section><p class="note">' + limitations + '</p><p><small>FAR: accepted impostor attempts. FRR: rejected genuine attempts. HTER: mean of FAR and FRR. Rates pool fixed-threshold decisions; AUC is averaged across folds. Initial models are untrained controls with development-calibrated thresholds.</small></p></main></html>'
    observations = observed_results(static, sequential)
    page = page.replace("<section><h2>Static", "<section><h2>Observed results</h2>" + "".join("<p>" + html.escape(x) + "</p>" for x in observations) + "</section><section><h2>Static", 1)
    (folder / "report.html").write_text(page, encoding="utf-8")
    lines = ["# EEGNet and graph verification", "", introduction, "", "## Static verification: 30-second queries", "",
             "| Method | Enrollment (s) | FAR | FRR | HTER | Mean fold AUC |", "|---|---:|---:|---:|---:|---:|"]
    lines = lines[:4] + ["## Observed results", "", "\n\n".join(observations), ""] + lines[4:]
    for row in primary.itertuples():
        lines.append(f"| {row.method} | {row.enrollment_seconds} | {row.FAR:.2%} | {row.FRR:.2%} | {row.HTER:.2%} | {row.mean_fold_AUC:.4f} |")
    lines += ["", "## Sequential verification: 30-second queries", "", explanation, "",
              "| Method | Enrollment (s) | Policy | FAR | FRR | HTER | Updates | Impostor updates |",
              "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in adaptive.itertuples():
        lines.append(f"| {row.method} | {row.enrollment_seconds} | {row.policy} | {row.FAR:.2%} | {row.FRR:.2%} | {row.HTER:.2%} | {row.updates} | {row.impostor_updates} |")
    lines += ["", "## Limits", "", limitations, "", "See the CSV files for all query durations and per-fold metrics."]
    (folder / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    build_demo(root, streams)
    save_json(folder / "validation.json", {"status": "passed", "protocol_sha256": p["protocol_sha256"],
              "folds": 6, "trained_models": 12, "static_metric_rows": len(metrics),
              "sequential_decisions_replayed": len(streams),
              "checks": ["all-model test freeze", "development thresholds", "static decisions and metrics",
                         "exact sequential replay without labels", "update counts and drift"]})
    print(f"Report complete: {len(streams)} sequential decisions independently replayed", flush=True)
