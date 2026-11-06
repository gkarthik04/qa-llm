"""Central configuration: model names, paths and default sizes.

Every value can be overridden with an environment variable of the same name
prefixed with ``QA_`` (e.g. ``QA_GEN_MODEL=google/flan-t5-large``).
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"QA_{name}", default)


# Models
BASE_READER = _env("BASE_READER", "distilbert-base-cased-distilled-squad")
FINETUNED_READER_DIR = Path(_env("FINETUNED_READER_DIR", str(ROOT / "models" / "distilbert-squad-finetuned")))
GEN_MODEL = _env("GEN_MODEL", "google/flan-t5-base")
EMBED_MODEL = _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Paths
INDEX_DIR = Path(_env("INDEX_DIR", str(ROOT / "index")))
RESULTS_DIR = Path(_env("RESULTS_DIR", str(ROOT / "results")))

# Data
DATASET = _env("DATASET", "rajpurkar/squad")
# The retrieval corpus is the set of unique SQuAD validation passages; the
# evaluation questions come from the same split (open-book over a known corpus).
CORPUS_SPLIT = _env("CORPUS_SPLIT", "validation")

# Tokenisation for the extractive reader
MAX_LENGTH = int(_env("MAX_LENGTH", "384"))
DOC_STRIDE = int(_env("DOC_STRIDE", "128"))
MAX_ANSWER_TOKENS = int(_env("MAX_ANSWER_TOKENS", "30"))

SEED = int(_env("SEED", "42"))


def cpu_name() -> str:
    """Human-readable CPU name for result logs."""
    import platform
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def extractive_reader_path() -> str:
    """Prefer our fine-tuned checkpoint, fall back to the public base model."""
    if (FINETUNED_READER_DIR / "config.json").exists():
        return str(FINETUNED_READER_DIR)
    return BASE_READER
