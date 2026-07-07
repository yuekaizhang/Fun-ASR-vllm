import ast
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _sampling_param_kwargs(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "SamplingParams":
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords if kw.arg}
            calls.append(kwargs)
    return calls


class VllmSamplingParamsTest(unittest.TestCase):
    def test_triton_wrapper_uses_deterministic_asr_decoding(self):
        calls = _sampling_param_kwargs(
            REPO_ROOT / "triton_server/model_repo_funasr/funasr/1/funasr_wrapper.py"
        )

        self.assertTrue(calls)
        self.assertEqual(calls[0].get("temperature"), 0.0)
        self.assertEqual(calls[0].get("top_p"), 1.0)
        self.assertTrue(calls[0].get("skip_special_tokens"))

    def test_offline_infer_uses_deterministic_asr_decoding(self):
        calls = _sampling_param_kwargs(REPO_ROOT / "infer.py")

        self.assertTrue(calls)
        self.assertEqual(calls[0].get("temperature"), 0.0)
        self.assertEqual(calls[0].get("top_p"), 1.0)
        self.assertTrue(calls[0].get("skip_special_tokens"))
