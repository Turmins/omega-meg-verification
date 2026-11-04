# Public demo and trained-weight release review

The educational purpose is explicit. It does not by itself establish redistribution
rights for inputs or learned artifacts from the supplied OMEGA export.

The [official OMEGA access page](https://www.mcgill.ca/bic/neuroinformatics/omega)
describes controlled access and prohibits redistributing the data. The page does
not explicitly resolve distribution of trained checkpoints or per-participant
biometric score matrices. The local export identifies itself as OMEGA but does not document the exact upstream release or accepted agreement. Its README/protocol contain no additional grant covering these derivative artifacts. The official page is a reference for the named source, not proof of the exact local export terms. This is an unresolved permission question, not a conclusion
that all learned weights are prohibited. No raw recordings are proposed for release.

The original V1 replay contains 60 participant pseudonyms, 7,200 individual
comparisons and 14,400 method decisions. Renaming those pseudonyms would still
leave individual biometric score records. It remains local under Git-ignored
`local_artifacts/verification_v1/report/`, unchanged. It is not bundled in the
public demo or Git history.

The public `demo/index.html` is newly built from the three already included
aggregate CSVs and handcrafted synthetic score/threshold examples. It contains
no participant records, actual DEV thresholds, trained weights, signals or network
dependencies. The real aggregate results and illustrative decisions are clearly
separated. It supports V1/V2 static/stream, model, enrollment, duration and policy
selectors, plus genuine/impostor/false-accept/false-reject examples. It does not run
neural inference and is not a replacement for the historical per-participant replay.

Open this HTML directly after cloning/downloading the repository; Python and the
private archive are not required. A GitHub source-file view does not itself run
the HTML. No GitHub Pages deployment is configured or performed.

Trained checkpoints are prepared only in the ignored local candidate directory.
The review covers pinned source hashes, tensor-only content, finite tensor values,
strict loading into the matching unchanged model architecture, text ID/path patterns
and actual file sizes. Passing these checks is not proof against memorization or
other participant information learned in biometric representations.

After applicable redistribution permission is confirmed, all 18 best checkpoints
could fit normal Git: the largest is 419,559 bytes and total size is 3,300,786 bytes.
GitHub's [documented file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
do not require LFS for files this small. No optimizer states, raw MEG, participant
templates, session maps or individual score tables are part of the candidate package.
No weights have been staged or committed while this review is pending.

The separate license decision for project/reference code remains outstanding.
Nothing has been uploaded or published to GitHub.
