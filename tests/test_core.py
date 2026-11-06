import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch


ENGINE = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_models_cli_uses_stdlib_and_reports_seven_exact_models(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'HP2_all_vocals.pth').write_bytes(b'fixture')
            completed = subprocess.run(
                [sys.executable, '-S', '-m', 'audio_vocal_engine', 'models', '--model-root', folder],
                cwd=ENGINE, text=True, encoding='utf-8', capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            models = json.loads(completed.stdout)['models']
            self.assertEqual([m['id'] for m in models], [
                'HP2_all_vocals', 'HP5_only_main_vocal',
                'model_bs_roformer_ep_317_sdr_12.9755', 'onnx_dereverb_By_FoxJoy',
                'VR-DeEchoAggressive', 'VR-DeEchoDeReverb', 'VR-DeEchoNormal',
            ])
            self.assertFalse(models[0]['available'])
            self.assertEqual(models[0]['status'], 'invalid')
            self.assertFalse(models[1]['available'])
            self.assertEqual(models[0]['parameters']['aggressiveness']['default'], 10)
            self.assertEqual(models[2]['parameters'], {})
            self.assertEqual(models[3]['parameters'], {})
            self.assertEqual(models[3]['path'], str(root / 'onnx_dereverb_By_FoxJoy' / 'vocals.onnx'))

    def test_independent_core_is_present(self):
        # These modules must be importable before any heavy runtime is installed.
        self.assertIsNotNone(importlib.util.find_spec('audio_vocal_engine.catalog'))
        self.assertIsNotNone(importlib.util.find_spec('audio_vocal_engine.job'))
        self.assertIsNotNone(importlib.util.find_spec('audio_vocal_engine.locking'))

    def test_vr_and_deecho_route_outputs_by_signature(self):
        from audio_vocal_engine.adapters import NativeAdapter

        class NativeWriter:
            def _path_audio_(self, source, **kwargs):
                self.kwargs = kwargs
                for key in ('vocal_root', 'ins_root'):
                    target = Path(kwargs[key])
                    target.mkdir(parents=True, exist_ok=True)
                    write_wav(target / 'intentionally_misleading_name.wav')

        with tempfile.TemporaryDirectory() as folder:
            for kind, primary in [('vr', 'vocals'), ('deecho', 'cleaned')]:
                native = NativeWriter()
                outputs = NativeAdapter(native, kind).process(Path('input.wav'), Path(folder) / kind)
                self.assertEqual(outputs[0]['role'], primary)
                self.assertEqual(Path(outputs[0]['path']).parent.name, primary)
                expected = 'ins_root' if kind == 'deecho' else 'vocal_root'
                self.assertEqual(Path(native.kwargs[expected]).name, primary)

    def test_completed_item_checkpoint_exists_before_event_and_accepts_more_than_12(self):
        from audio_vocal_engine.job import run_job

        class Backend:
            runtime = {'device': 'cpu', 'precision': 'fp32'}

            def process(self, source, output_dir):
                path = output_dir / 'vocals.wav'
                write_wav(path)
                return [{'role': 'vocals', 'path': str(path)}]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'original.wav'
            write_wav(source)
            source_bytes = source.read_bytes()
            weights = root / 'models'
            weights.mkdir()
            (weights / 'HP2_all_vocals.pth').write_bytes(b'fixture')
            request = request_fixture(source, weights, root / 'outputs')
            request['items'] = [dict(request['items'][0], id=str(i)) for i in range(13)]
            result_path = root / 'result.json'
            events = []

            def event(value):
                events.append(value)
                if value.get('item'):
                    saved = json.loads(result_path.read_text(encoding='utf-8'))
                    self.assertEqual(saved['items'][-1], value['item'])
                    self.assertEqual(saved['parameters'], {'aggressiveness': 10})
                    self.assertEqual(saved['model']['id'], request['model_id'])

            with patch('audio_vocal_engine.catalog.load_manifest', return_value=fixture_manifest(weights)), \
                 patch('audio_vocal_engine.job.prepare_input', side_effect=lambda src, dst: src):
                result = run_job(request, result_path, event, backend_factory=lambda *args: Backend(), state_dir=root / 'state')
            self.assertEqual(len(result['items']), 13)
            paths = {i['outputs'][0]['path'] for i in result['items']}
            self.assertEqual(len(paths), 13)
            self.assertEqual(result['items'][0]['outputs'][0]['source_start'], 2.5)
            self.assertEqual(source.read_bytes(), source_bytes)
            self.assertEqual(result['items'][0]['runtime'], Backend.runtime)
            self.assertEqual(result['items'][0]['model'], result['model'])
            self.assertEqual(events[-1]['stage'], 'completed')
            self.assertEqual(events[-1]['progress'], 100)

    def test_verified_weight_and_config_change_the_version_and_missing_manifest_is_safe(self):
        from audio_vocal_engine.catalog import get_model, list_models, load_manifest
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'HP2_all_vocals.pth').write_bytes(b'fixture')
            config = root / 'config.json'
            config.write_text('{"version": 1}', encoding='utf-8')
            with patch('audio_vocal_engine.catalog.load_manifest', return_value=fixture_manifest(root)), \
                 patch('audio_vocal_engine.catalog.model_config_path', return_value=config):
                first = get_model(root, 'HP2_all_vocals')
                self.assertTrue(first['available'])
                config.write_text('{"version": 2}', encoding='utf-8')
                second = get_model(root, 'HP2_all_vocals')
                self.assertNotEqual(first['version'], second['version'])
                (root / 'HP2_all_vocals.pth').write_bytes(b'changed')
                changed = get_model(root, 'HP2_all_vocals')
                self.assertFalse(changed['available'])
                self.assertNotEqual(second['version'], changed['version'])
            with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
                self.assertEqual(load_manifest(), {})
                self.assertEqual(len(list_models(root)), 7)

    def test_waiting_happens_before_load_and_null_end_is_measured(self):
        from audio_vocal_engine.job import run_job
        from audio_vocal_engine.locking import EngineLock

        class Backend:
            runtime = {'device': 'cpu', 'precision': 'fp32'}

            def process(self, source, output_dir):
                path = output_dir / 'vocals.wav'
                write_wav(path)
                return [{'role': 'vocals', 'path': str(path)}]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'input.wav'
            write_wav(source)
            weights = root / 'weights'
            weights.mkdir()
            (weights / 'HP2_all_vocals.pth').write_bytes(b'fixture')
            request = request_fixture(source, weights, root / 'outputs')
            request['items'][0]['source_end'] = None
            owner = EngineLock(root / 'state/engine.lock')
            owner.acquire()
            loaded = []
            stages = []

            def event(value):
                stages.append(value['stage'])
                if value['stage'] == 'waiting':
                    self.assertEqual(loaded, [])
                    owner.release()

            def factory(*args):
                loaded.append(True)
                return Backend()

            try:
                with patch('audio_vocal_engine.catalog.load_manifest', return_value=fixture_manifest(weights)), \
                     patch('audio_vocal_engine.job.prepare_input', side_effect=lambda src, dst: src):
                    result = run_job(request, root / 'result.json', event, factory, root / 'state')
            finally:
                owner.release()
            self.assertEqual(stages[0:2], ['waiting', 'loading'])
            output = result['items'][0]['outputs'][0]
            self.assertAlmostEqual(output['source_end'], 2.5 + output['duration'])

    def test_bad_output_fails_only_its_item_and_completed_output_is_checkpointed(self):
        from audio_vocal_engine.job import run_job

        class Backend:
            runtime = {'device': 'cpu', 'precision': 'fp32'}
            calls = 0

            def process(self, source, output_dir):
                self.calls += 1
                path = output_dir / 'vocals.wav'
                write_wav(path, frames=44100 if self.calls == 1 else 100)
                return [{'role': 'vocals', 'path': str(path)}]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'original.wav'
            write_wav(source)
            weights = root / 'weights'
            weights.mkdir()
            (weights / 'HP2_all_vocals.pth').write_bytes(b'fixture')
            request = request_fixture(source, weights, root / 'outputs')
            request['items'].append(dict(request['items'][0], id='two'))
            result_path = root / 'result.json'
            with patch('audio_vocal_engine.catalog.load_manifest', return_value=fixture_manifest(weights)), \
                 patch('audio_vocal_engine.job.prepare_input', side_effect=lambda src, dst: src):
                result = run_job(request, result_path, lambda event: None, lambda *args: Backend(), root / 'state')
            self.assertIn('duration', result['items'][0]['error'])
            self.assertEqual(result['items'][0]['outputs'], [])
            self.assertNotIn('error', result['items'][1])
            self.assertEqual(json.loads(result_path.read_text(encoding='utf-8'))['items'], result['items'])

    def test_invalid_request_cli_writes_error_result(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            request_path, result_path = root / 'request.json', root / 'result.json'
            request_path.write_text('{}', encoding='utf-8')
            completed = subprocess.run([sys.executable, '-S', '-m', 'audio_vocal_engine', 'run', str(request_path), str(result_path)],
                                       cwd=ENGINE, capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(completed.returncode, 1)
            self.assertIn('error', json.loads(result_path.read_text(encoding='utf-8')))
            self.assertEqual(json.loads(completed.stdout)['stage'], 'failed')

    def test_native_paths_stay_short_and_outputs_publish_to_long_caller_directory(self):
        from audio_vocal_engine.adapters import NativeAdapter
        from audio_vocal_engine.job import run_job

        native_paths = []

        class NativeWriter:
            def _path_audio_(self, source, **kwargs):
                native_paths.append(source)
                for key in ('vocal_root', 'ins_root'):
                    path = Path(kwargs[key]) / 'native.wav'
                    native_paths.append(str(path))
                    if len(str(path)) >= 200:
                        raise RuntimeError('Native library requires a short path.')
                    write_wav(path)

        class Backend:
            runtime = {'device': 'cpu', 'precision': 'fp32'}

            def process(self, source, output_dir):
                return NativeAdapter(NativeWriter(), 'vr').process(source, output_dir)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'original.wav'
            write_wav(source)
            weights = root / 'weights'
            weights.mkdir()
            (weights / 'HP2_all_vocals.pth').write_bytes(b'fixture')
            output_dir = root.joinpath(*(['caller-' + 'x' * 50] * 4))
            request = request_fixture(source, weights, output_dir)

            def prepare(src, destination):
                shutil.copyfile(src, destination)
                return destination

            try:
                with patch('audio_vocal_engine.catalog.load_manifest', return_value=fixture_manifest(weights)), \
                     patch('audio_vocal_engine.job.prepare_input', side_effect=prepare):
                    result = run_job(request, root / 'result.json', lambda event: None, lambda *args: Backend(), root / 'state')
                self.assertNotIn('error', result, result)
                self.assertTrue(all(len(path) < 200 for path in native_paths))
                for output in result['items'][0]['outputs']:
                    path = Path(output['path'])
                    self.assertTrue(path.is_relative_to(output_dir))
                    self.assertTrue(Path(filesystem_test_path(path)).is_file())
                    self.assertLessEqual(len(path.name), 32)
                self.assertEqual(list((root / 'state/work').iterdir()), [])
            finally:
                # This is exactly the new TemporaryDirectory owned by this test.
                self.assertEqual(root.resolve(), Path(folder).resolve())
                shutil.rmtree(filesystem_test_path(root))

    def test_cancel_cleans_private_work_and_preserves_completed_checkpoint(self):
        from audio_vocal_engine.job import run_job

        class Backend:
            runtime = {'device': 'cpu', 'precision': 'fp32'}
            calls = 0

            def process(self, source, output_dir):
                self.calls += 1
                if self.calls == 2:
                    raise KeyboardInterrupt('cancelled')
                path = output_dir / 'native.wav'
                write_wav(path)
                return [{'role': 'vocals', 'path': str(path)}]

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'original.wav'
            write_wav(source)
            weights = root / 'weights'
            weights.mkdir()
            (weights / 'HP2_all_vocals.pth').write_bytes(b'fixture')
            request = request_fixture(source, weights, root / 'outputs')
            request['items'].append(dict(request['items'][0], id='two'))
            result_path = root / 'result.json'
            with patch('audio_vocal_engine.catalog.load_manifest', return_value=fixture_manifest(weights)), \
                 patch('audio_vocal_engine.job.prepare_input', side_effect=lambda src, dst: src), \
                 self.assertRaises(KeyboardInterrupt):
                run_job(request, result_path, lambda event: None, lambda *args: Backend(), root / 'state')
            saved = json.loads(result_path.read_text(encoding='utf-8'))
            self.assertEqual(len(saved['items']), 1)
            self.assertTrue(Path(saved['items'][0]['outputs'][0]['path']).is_file())
            self.assertEqual(list((root / 'state/work').iterdir()), [])

    def test_forced_exit_cleanup_matches_job_and_original_process_identity(self):
        from audio_vocal_engine.work import cleanup_work, process_identity
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = root / 'state'
            owner = subprocess.Popen(
                [getattr(sys, '_base_executable', sys.executable), '-c',
                 'from audio_vocal_engine.work import private_workspace\nimport sys\n'
                 'with private_workspace(sys.argv[1], sys.argv[2]) as path:\n'
                 ' print(path, flush=True)\n sys.stdin.read()\n', 'forced-exit', str(state)],
                cwd=ENGINE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8',
            )
            try:
                workspace = Path(owner.stdout.readline().strip())
                self.assertTrue(workspace.is_dir())
                original = json.loads((workspace / 'owner.json').read_text(encoding='utf-8'))
                self.assertEqual(process_identity(owner.pid), original['create_time'])
                self.assertEqual(cleanup_work('forced-exit', state), [])
                owner.terminate()
                owner.wait(timeout=5)
                self.assertIsNone(process_identity(owner.pid))
                self.assertEqual(cleanup_work('another-job', state), [])
                foreign = state / 'work/foreign'
                foreign.mkdir()
                ambiguous = state / 'work/w-ffffffffffff'
                ambiguous.mkdir()
                incomplete_owner = dict(original, directory=ambiguous.name)
                incomplete_owner.pop('engine_root')
                (ambiguous / 'owner.json').write_text(json.dumps(incomplete_owner), encoding='utf-8')
                # A recycled PID also must not make the original owner look alive.
                with patch('audio_vocal_engine.work.process_identity', return_value='different-create-time'):
                    self.assertEqual(cleanup_work('forced-exit', state), [str(workspace)])
                self.assertFalse(workspace.exists())
                self.assertTrue(foreign.is_dir())
                self.assertTrue(ambiguous.is_dir())
            finally:
                if owner.poll() is None:
                    owner.terminate()
                    owner.wait(timeout=5)
                owner.stdin.close()
                owner.stdout.close()

    def test_lock_releases_when_owner_process_exits(self):
        from audio_vocal_engine.locking import EngineLock
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'queue.lock'
            owner = subprocess.Popen(
                [getattr(sys, '_base_executable', sys.executable), '-c',
                 'from audio_vocal_engine.locking import EngineLock; import sys; '
                 'lock = EngineLock(sys.argv[1]); lock.acquire(); print("locked", flush=True); sys.stdin.read()', str(path)],
                cwd=ENGINE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            )
            try:
                self.assertEqual(owner.stdout.readline().strip(), 'locked')
                contender = EngineLock(path)
                self.assertFalse(contender.try_acquire())
                owner.terminate()
                owner.wait(timeout=5)
                self.assertTrue(contender.try_acquire())
                contender.release()
            finally:
                if owner.poll() is None:
                    owner.terminate()
                    owner.wait(timeout=5)
                owner.stdin.close()
                owner.stdout.close()


def write_wav(path, frames=100):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(44100)
        output.writeframes(b'\x00' * frames * 4)


def filesystem_test_path(path):
    value = str(Path(path).resolve())
    return '\\\\?\\' + value if os.name == 'nt' and not value.startswith('\\\\?\\') else value


def fixture_manifest(root):
    path = root / 'HP2_all_vocals.pth'
    return {'HP2_all_vocals': {'path': path.name, 'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}


def request_fixture(source, weights, output_dir):
    with wave.open(str(source), 'rb') as audio:
        duration = audio.getnframes() / audio.getframerate()
    return {
        'schema_version': 1, 'job_id': 'core-test', 'caller': 'unit-test',
        'model_root': str(weights), 'model_id': 'HP2_all_vocals',
        'device': 'cpu', 'precision': 'fp32', 'parameters': {'aggressiveness': 10},
        'items': [{'id': 'one', 'clip_id': 'clip', 'input_version_id': 'original',
                   'input': str(source), 'output_dir': str(output_dir),
                   'source_start': 2.5, 'source_end': 2.5 + duration}],
    }


if __name__ == '__main__':
    unittest.main()
