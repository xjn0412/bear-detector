"""Process polling must work without pidfd support and ignore exited zombies."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location(
    'handoff', Path(__file__).resolve().parents[1] / 'tools/optimize_v2_at_checkpoint.py')
handoff = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handoff)


class PollingTests(unittest.TestCase):
    def test_zombie_is_exited_without_wait(self):
        process = Mock()
        process.is_running.return_value = True
        process.status.return_value = handoff.psutil.STATUS_ZOMBIE
        process.wait.side_effect = OSError(22, 'pidfd unsupported')
        self.assertEqual(handoff.wait_until_exited([process], 0), [])
        process.wait.assert_not_called()

    def test_running_process_retained_at_deadline(self):
        process = Mock()
        process.is_running.return_value = True
        process.status.return_value = handoff.psutil.STATUS_RUNNING
        self.assertEqual(handoff.wait_until_exited([process], 0), [process])

    def test_process_exiting_during_poll(self):
        process = Mock()
        process.is_running.side_effect = [True, False]
        process.status.return_value = handoff.psutil.STATUS_RUNNING
        with patch.object(handoff.time, 'sleep'):
            self.assertEqual(handoff.wait_until_exited([process], 5), [])


if __name__ == '__main__':
    unittest.main()
