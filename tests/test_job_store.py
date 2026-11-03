import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database.jobs import JobStore


class JobStoreTests(unittest.TestCase):
    def test_registration_is_idempotent_and_changed_source_is_new_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "INPUT"
            input_dir.mkdir()
            source = input_dir / "photo.jpg"
            source.write_bytes(b"first")
            store = JobStore(root / "jobs.sqlite3")

            job_id, created = store.register(source)
            same_id, duplicate_created = store.register(source)
            self.assertTrue(created)
            self.assertFalse(duplicate_created)
            self.assertEqual(job_id, same_id)

            source.write_bytes(b"updated content")
            changed_id, changed_created = store.register(source)
            self.assertTrue(changed_created)
            self.assertNotEqual(job_id, changed_id)
            self.assertEqual(len(store.pending(input_dir)), 2)

    def test_recovers_interrupted_jobs_and_retries_failed_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "INPUT"
            input_dir.mkdir()
            source = input_dir / "photo.jpg"
            source.write_bytes(b"photo")
            database = root / "jobs.sqlite3"
            store = JobStore(database)
            job_id, _ = store.register(source)
            store.set_status(job_id, "in_progress")

            recovered = JobStore(database)
            self.assertEqual(recovered.pending(input_dir), [(job_id, source.resolve())])
            recovered.set_status(job_id, "failed", error="test failure")
            self.assertEqual(recovered.retry_failed(input_dir), 1)
            self.assertEqual(recovered.counts(input_dir)["pending"], 1)

    def test_counts_only_requested_input_folder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_a = root / "INPUT_A"
            input_b = root / "INPUT_B"
            input_a.mkdir()
            input_b.mkdir()
            (input_a / "a.jpg").write_bytes(b"a")
            (input_b / "b.jpg").write_bytes(b"b")
            store = JobStore(root / "jobs.sqlite3")
            store.register(input_a / "a.jpg")
            store.register(input_b / "b.jpg")
            self.assertEqual(store.counts(input_a), {"pending": 1})


if __name__ == "__main__":
    unittest.main()
