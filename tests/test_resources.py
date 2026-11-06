"""Resource-selection checks without importing real inference runtimes."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


GIB = 1024 ** 3


def mocked_runtimes(free_bytes):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True, mem_get_info=Mock(return_value=(free_bytes, 10 * GIB))),
        get_num_threads=Mock(return_value=48), set_num_threads=Mock(), device=lambda name: name,
    )
    ort = SimpleNamespace(get_available_providers=lambda: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return torch, ort


class ResourceTests(unittest.TestCase):
    def test_low_memory_auto_uses_cpu_and_explicit_cuda_refuses(self):
        from audio_vocal_engine.adapters import resolve_runtime
        torch, ort = mocked_runtimes(3 * GIB)
        with patch.dict(sys.modules, {'torch': torch, 'onnxruntime': ort}):
            runtime = resolve_runtime('auto', 'auto', 'mdx', 'onnx_dereverb_By_FoxJoy')
            self.assertEqual(runtime['device'], 'cpu')
            self.assertEqual(runtime['precision'], 'fp32')
            self.assertEqual(runtime['free_gpu_bytes'], 3 * GIB)
            self.assertEqual(runtime['required_free_gpu_bytes'], 4 * GIB)
            self.assertTrue(runtime['selection_reason'])
            torch.set_num_threads.assert_called_with(4)
            with self.assertRaisesRegex(RuntimeError, '4.0'):
                resolve_runtime('cuda', 'fp32', 'mdx', 'onnx_dereverb_By_FoxJoy')

    def test_model_budgets_and_explicit_precision_are_preserved(self):
        from audio_vocal_engine.adapters import resolve_runtime
        torch, ort = mocked_runtimes(int(2.5 * GIB))
        with patch.dict(sys.modules, {'torch': torch, 'onnxruntime': ort}):
            vr = resolve_runtime('auto', 'fp32', 'vr', 'HP2_all_vocals')
            self.assertEqual((vr['device'], vr['precision']), ('cuda', 'fp32'))
            self.assertEqual(vr['required_free_gpu_bytes'], 2 * GIB)
            deecho = resolve_runtime('auto', 'auto', 'deecho', 'VR-DeEchoDeReverb')
            self.assertEqual(deecho['device'], 'cpu')
            self.assertEqual(deecho['required_free_gpu_bytes'], 3 * GIB)
            roformer = resolve_runtime('auto', 'auto', 'roformer', 'model_bs_roformer_ep_317_sdr_12.9755')
            self.assertEqual(roformer['device'], 'cpu')
            self.assertEqual(roformer['required_free_gpu_bytes'], 4 * GIB)
            with self.assertRaisesRegex(ValueError, 'fp16'):
                resolve_runtime('auto', 'fp16', 'roformer', 'model_bs_roformer_ep_317_sdr_12.9755')
            torch.cuda.mem_get_info.return_value = (8 * GIB, 10 * GIB)
            selected = resolve_runtime('auto', 'auto', 'roformer', 'model_bs_roformer_ep_317_sdr_12.9755')
            self.assertEqual((selected['device'], selected['precision']), ('cuda', 'fp16'))

    def test_unknown_memory_is_conservative_and_cpu_request_does_not_query_cuda(self):
        from audio_vocal_engine.adapters import resolve_runtime
        torch, ort = mocked_runtimes(3 * GIB)
        torch.cuda.mem_get_info.side_effect = RuntimeError('memory query failed')
        with patch.dict(sys.modules, {'torch': torch, 'onnxruntime': ort}):
            runtime = resolve_runtime('auto', 'auto', 'vr', 'HP2_all_vocals')
            self.assertEqual(runtime['device'], 'cpu')
            self.assertIsNone(runtime['free_gpu_bytes'])
            with self.assertRaises(RuntimeError):
                resolve_runtime('cuda', 'fp32', 'vr', 'HP2_all_vocals')
            torch.cuda.mem_get_info.reset_mock()
            resolve_runtime('cpu', 'fp32', 'vr', 'HP2_all_vocals')
            torch.cuda.mem_get_info.assert_not_called()

    def test_onnx_session_uses_four_intra_op_threads(self):
        torch, ort = mocked_runtimes(8 * GIB)
        ort.SessionOptions = Mock(side_effect=lambda: SimpleNamespace(intra_op_num_threads=0))
        ort.InferenceSession = Mock(return_value=object())
        modules = {'torch': torch, 'onnxruntime': ort, 'librosa': SimpleNamespace(),
                   'numpy': SimpleNamespace(), 'soundfile': SimpleNamespace(), 'tqdm': SimpleNamespace(tqdm=Mock())}
        path = Path(__file__).resolve().parents[1] / 'audio_vocal_engine/vendor/mdxnet.py'
        spec = importlib.util.spec_from_file_location('isolated_mdx_resource_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
            with patch.object(module, 'get_models', return_value=SimpleNamespace(target_name='vocals')):
                module.Predictor(SimpleNamespace(onnx='/models/foxjoy', providers=['CPUExecutionProvider'], dim_f=3072, dim_t=9, n_fft=6144))
        self.assertEqual(ort.InferenceSession.call_args.kwargs['sess_options'].intra_op_num_threads, 4)


if __name__ == '__main__':
    unittest.main()
