""" Building Vocabulary """

from pathlib import Path
import sys

from pipeline_exam.src.vocabulary_building import build_step02_parser, run_step02

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def main() -> None:
    parser = build_step02_parser(_REPO_ROOT)
    args = parser.parse_args()
    run_step02(args)

if __name__ == "__main__":
    main()