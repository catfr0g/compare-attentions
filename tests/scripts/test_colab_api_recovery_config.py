"""Actual export07 parser/layout regression, offline and stopped before native restore."""

# pylint: disable=import-outside-toplevel

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXPORT = Path("E:/HermesArtifacts/compare-attention/colab-k64-inspect-07")
ARCHIVE = Path("E:/HermesArtifacts/compare-attention/colab-k64-export-07/colab-package.zip")


def test_generated_native_config_prefix_matches_actual_export07(tmp_path):
    """Use the verified original wheel, YAML, data and real tyro/Pydantic parser."""
    if not EXPORT.is_dir() or not ARCHIVE.is_file():
        pytest.skip("requires local SHA-verified export07 artifacts (no download/install)")
    program = r"""
import ast
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
root, package, archive = map(Path, sys.argv[1:])
sys.path.insert(0, str(root / 'scripts'))
import colab_api_recovery as recovery

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

assert sha(archive) == recovery.ARCHIVE_SHA
assert sha(package / 'manifest.json') == recovery.MANIFEST_SHA
# Import the original extracted tools, never a checkout substitute.
sys.path.insert(0, str(package / 'tools'))
import colab_package
assert Path(colab_package.__file__).parent == package / 'tools'
manifest = colab_package.verify_directory(package)
wheel = package / 'release' / colab_package.WHEEL
assert sha(wheel) == 'f5261d6b6c79483a96fc3c7ac0cbd6cb3e66b948968646bff88f0dcb085a1274'
sys.path.insert(0, str(wheel))
from compare_attention.cli import train
from compare_attention.train.config import TrainConfig
assert str(wheel) in train.__file__
config_path = package / 'release' / colab_package.CONFIGS['deltanet']
assert not (package / 'release/deltanet.yaml').exists()
# Real tyro parsing resolves postponed annotations in the original module.
original_argv = ['native_program', '--not-a-training-flag']
sys.argv = original_argv
expected_args = train.parse_train_args(
    ['--config', str(config_path), '--data', str(package / 'data')])
excluded = {'config', 'kv', 'result_dir', 'resume_from', 'tracker', 'hf_repo'}
expected = TrainConfig.model_validate(expected_args.model_dump(exclude=excluded))
context = {'paths': {'package': str(package), 'snapshot': 'UNOPENED.pt', 'root': 'UNWRITTEN'},
           'spec': recovery.RecoverySpec('runtimes/local-fixture', 'local-fixture-run').value()}
source = recovery.native_program(context)
tree = ast.parse(source)
# Keep every generated parser statement verbatim; stop at the first native call.
boundary = next(i for i, node in enumerate(tree.body)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'validate_native')
prefix = ast.Module(body=tree.body[:boundary], type_ignores=[])
namespace = {'__name__': '__main__'}
exec(compile(prefix, '<actual-generated-config-prefix>', 'exec'), namespace)
captured = []
def boundary_marker(config, snapshot, spec):
    captured.append((config, snapshot, spec))
namespace['validate_native'] = boundary_marker
exec(compile(ast.Module(body=[tree.body[boundary]], type_ignores=[]),
             '<validate-native-no-op-boundary>', 'exec'), namespace)
assert len(captured) == 1
actual, snapshot, spec = captured[0]
assert actual.model_dump(mode='json') == expected.model_dump(mode='json')
assert actual.data == package / 'data'
assert actual.schedule.train_flops == 10000000000000000
assert actual.schedule.dtype == 'float16'
assert actual.schedule.seed == 42
assert actual.model.attention == 'deltanet'
assert actual.schedule.amp_max_scale == 128.0
assert actual.dataset.num_kv_pairs == 64
assert snapshot == Path('UNOPENED.pt') and spec == context['spec']
assert sys.argv is original_argv and sys.argv == ['native_program', '--not-a-training-flag']
assert namespace['__name__'] == '__main__'
print(json.dumps({'boundary_calls': len(captured), 'config': actual.model_dump(mode='json'),
                  'config_path': str(config_path), 'wheel_sha256': sha(wheel),
                  'manifest_sha256': sha(package / 'manifest.json'),
                  'generated_program_sha256': hashlib.sha256(source.encode()).hexdigest(),
                  'argv_preserved': True, 'original_tyro_and_pydantic': True}, sort_keys=True))
"""
    # Preserve Windows DLL/user discovery essentials but do not inherit service secrets.
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {"PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "USERNAME", "USERPROFILE", "HOME"}
    }
    env.update(
        CUDA_VISIBLE_DEVICES="-1",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
        PYTHON_DOTENV_DISABLED="1",
        PYTHONDONTWRITEBYTECODE="1",
        TORCHINDUCTOR_CACHE_DIR=str(tmp_path / "inductor"),
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(ROOT), str(EXPORT), str(ARCHIVE)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"boundary_calls": 1' in result.stdout
    print(result.stdout)
