""" Training """

from pathlib import Path
import sys

from pipeline_exam.src.training import build_step03_parser, run_step03

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main() -> None:
    parser = build_step03_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step03(args)

if __name__ == "__main__":
    main()