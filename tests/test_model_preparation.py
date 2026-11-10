from pathlib import Path
import runpy

from safetensors.torch import load_file
import torch


def test_weight_conversion_preserves_tensors_and_original(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/download_models.py"
    convert = runpy.run_path(str(script))["convert_blip"]
    state = {"weight": torch.randn(3, 4), "positions": torch.arange(4)}
    source = tmp_path / "upstream.bin"
    target = tmp_path / "model.safetensors"
    torch.save(state, source)
    assert convert(source, target) == len(state)
    assert source.is_file()
    restored = load_file(str(target))
    assert set(restored) == set(state)
    for name in state:
        assert torch.equal(state[name], restored[name])
