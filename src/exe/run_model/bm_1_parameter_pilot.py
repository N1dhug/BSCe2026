"""Run the bounded physical-parameter development pilot with runtime progress."""
import argparse
from pathlib import Path

from src.exe.run_model.progress import runtime_text
from src.validation.bm_1.parameter_optimization import run


def progress(label, method, candidates, started, completed, total):
    print(label, method, 'candidates', candidates,
          runtime_text(started, completed, total), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=600.)
    args = parser.parse_args()
    run(args.output, max_seconds=args.max_seconds, progress=progress)


if __name__ == '__main__':
    main()
