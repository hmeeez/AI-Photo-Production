import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.worker import BatchWorker
from database.jobs import JobStore


class BatchWorkerTests(unittest.TestCase):
    def test_batch_worker_processes_pending_job_and_persists_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "INPUT"
            input_dir.mkdir()
            source = input_dir / "sample.png"
            Image.new("RGB", (80, 60), (100, 120, 140)).save(source)
            store = JobStore(root / "jobs.sqlite3")
            store.register(source)
            worker = BatchWorker(
                input_dir,
                root / "OUTPUT",
                {"enable_print_output": False},
                store,
                monitor_mode=False,
            )
            results: list[str] = []
            worker.item_done.connect(lambda _name, status: results.append(status))

            worker.run()

            self.assertEqual(results, ["completed"])
            self.assertEqual(store.counts(input_dir)["completed"], 1)


if __name__ == "__main__":
    unittest.main()
