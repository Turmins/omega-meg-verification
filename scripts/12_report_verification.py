"""Validate saved V1 results and create an English report and saved replay."""
import argparse
import os
from _bootstrap import ROOT

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='outputs/verification_v1')
    args = parser.parse_args()
    for name in ('TEMP', 'TMP', 'MPLCONFIGDIR'):
        path = ROOT / 'outputs/runtime' / name.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    from omega_dl.verification.reporting import build_report
    print(build_report(ROOT / args.out))

if __name__ == '__main__':
    main()
