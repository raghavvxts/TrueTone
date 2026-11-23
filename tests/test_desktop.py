"""Check wrapper metadata, local project binding, and build commands without compiling."""
import importlib.util
from pathlib import Path
import plistlib
import tempfile
import unittest
from unittest.mock import patch


class DesktopTests(unittest.TestCase):
    def test_bundle_points_to_project_and_declares_microphone(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/build_desktop_app.py'
        spec = importlib.util.spec_from_file_location('desktop_builder', source)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as folder, patch.object(builder.subprocess, 'run') as run:
            destination = Path(folder) / 'TrueTone.app'
            builder.build(destination)
            contents = destination / 'Contents'
            info = plistlib.loads((contents / 'Info.plist').read_bytes())
            self.assertEqual(info['CFBundleExecutable'], 'TrueTone')
            self.assertEqual(info['CFBundleIconFile'], 'TrueTone.icns')
            self.assertTrue((contents / 'Resources/TrueTone.icns').is_file())
            self.assertTrue((contents / 'Resources/logo.png').is_file())
            self.assertIn('locally', info['NSMicrophoneUsageDescription'])
            self.assertEqual((contents / 'Resources/project-path.txt').read_text(), str(builder.ROOT))
            self.assertEqual(run.call_count, 3)
            self.assertEqual(run.call_args_list[0].args[0][:2], ['xcrun', 'swiftc'])
            self.assertEqual(run.call_args_list[1].args[0][0], 'codesign')
            signing = run.call_args_list[1].args[0]
            entitlements = Path(signing[signing.index('--entitlements') + 1])
            access = plistlib.loads(entitlements.read_bytes())
            self.assertTrue(access['com.apple.security.device.audio-input'])
            self.assertTrue(access['com.apple.security.device.microphone'])
            self.assertNotIn('com.apple.security.device.camera', access)
            self.assertTrue(run.call_args_list[2].args[0][0].endswith('/lsregister'))
            self.assertEqual(run.call_args_list[2].args[0][1:], ['-f', str(destination)])

    def test_watchdog_exits_when_parent_disappears(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/run_desktop_server.py'
        spec = importlib.util.spec_from_file_location('desktop_server', source)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with patch.object(runner.os, 'getppid', side_effect=[42, 1]), \
             patch.object(runner.time, 'sleep') as sleep, \
             patch.object(runner.os, '_exit') as exit_process:
            runner.watch_parent(42)
            sleep.assert_called_once_with(1)
            exit_process.assert_called_once_with(0)
