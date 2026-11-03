# V1/V2 verification protocol

The question is whether a query matches the claimed person's reference, not which
identity wins a closed-set search. The supplied data contain 60 people with two
recordings each. Six deterministic identity-disjoint folds use 40 TRAIN, 10 DEV,
10 TEST identities; both recordings stay in the same partition and each person
is tested once. Enrollment and query use different sessions.

V1 enrollment is 240 seconds; 10/30/60-second queries start at 0/60/120/180 seconds.
Every query consists only of its own independently processed 10-second blocks.
The phase/coherency control and trained Siamese CNN are compared with an untrained
encoder. Checkpoint selection maximizes 30-second DEV AUC, earliest ties winning.

V2 uses EEGNet and graph encoders, 30/240-second enrollment and all nonoverlapping
10/30/60-second query intervals. Checkpoint selection uses 30/30-second DEV AUC.
Static all-pairs and balanced chronological stream results are separate evaluations.
Static, accepted-update and guarded-update policies do not retrain the encoder.

Acceptance thresholds minimize DEV abs(FAR-FRR), then HTER, then prefer the higher
threshold. Accept when score >= threshold. V2 update gates additionally target
at most 1% DEV impostor acceptance and cannot be below the acceptance threshold.
This target is not a guarantee for TEST. Checkpoints and calibrations are locked
before TEST evaluation of each run, and every loader retains its hash barriers.

FAR is accepted impostor attempts / impostor attempts; FRR is rejected genuine
attempts / genuine attempts; HTER=(FAR+FRR)/2; TAR=1-FRR. Decisions use each fold's
own DEV threshold. ROC-AUC and finite-candidate diagnostic EER are computed within
folds then averaged, not from a pooled cross-model ROC. TEST EER does not choose
the operating threshold. Similarity scores are not calibrated probabilities.

The dataset was already studied, comparisons reuse participants and intervals,
and training folds overlap. No naive independent-row CI or independent external
validation is claimed. V1/V2 percentages use different grids. There was one base
training seed per original series; this subset does not reproduce later DEV seed
studies. No new scientific training result is created by publication preparation.
