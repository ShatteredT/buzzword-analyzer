import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_data import prepare_dataset


def main() -> None:
    concrete_sample = (
        "Write unit tests in pytest with 85% coverage. "
        "Deploy Docker containers to AWS ECS using GitHub Actions. "
        "Reduce p95 API latency from 450 ms to 120 ms. "
        "Maintain PostgreSQL schemas and migration scripts. "
        "Implement rate limiting with Redis and token bucket. "
        "Review pull requests and fix regression bugs in Python. "
        "Document REST endpoints in OpenAPI 3.1 specification. "
        "Configure Prometheus alerts for CPU and memory usage."
    )
    buzzword_sample = (
        "Drive cross-functional synergy across dynamic global teams. "
        "Foster a culture of radical innovation and thought leadership. "
        "Deliver best-in-class holistic solutions with proactive mindset. "
        "Leverage cutting-edge paradigms to unlock seamless value. "
        "Champion agile transformation in fast-paced environments. "
        "Empower stakeholders through strategic alignment and vision. "
        "Cultivate high-impact collaboration with a growth mindset. "
        "Scale mission-critical excellence across evolving ecosystems."
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        c_file = tmp_path / "concrete.txt"
        b_file = tmp_path / "buzzwords.txt"
        out_dir = tmp_path / "processed"
        rep_dir = tmp_path / "reports"

        c_file.write_text(concrete_sample, encoding="utf-8")
        b_file.write_text(buzzword_sample, encoding="utf-8")

        stats = prepare_dataset(c_file, b_file, out_dir, rep_dir)
        assert stats["total_unique"] == 16, f"Expected 16 unique rows, got {stats['total_unique']}"
        assert (out_dir / "train.csv").exists()
        assert (out_dir / "val.csv").exists()
        assert (out_dir / "test.csv").exists()
        print("SYNTHETIC TEST PASSED:", stats)


if __name__ == "__main__":
    main()