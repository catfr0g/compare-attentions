"""Host-only stdlib recovery binding for the single approved export07 snapshot."""

# Exact JSON scalar types reject bool-as-int at the public status boundary.
# pylint: disable=unidiomatic-typecheck,too-many-boolean-expressions

import hashlib
import json
import re
from dataclasses import dataclass

ARCHIVE_SHA = "5e38fa71804e9c5d8095ea2ed048b9a97b32ef0e9050c78414f0d97d6b23b0ed"
MANIFEST_SHA = "194c694612ab12b79f68d5ded4b52ccbd53106943b1a1d012c1ca808323314f3"
SOURCE_SHA = "1beba22ca8f9897e0b7367e07f4a016559946f1183fc8413deff2490c6e85ef1"
DATA_SHA = "1c660aa5f54d88ea9e0c1ce4eba0c25084eee164c55ab6c1d80d5db14d3e59fb"
ORIGIN = "deltanet-k64-seed42-colab07b"

# Executed only in the already prepared frozen venv. No Task, trainer or telemetry.
NATIVE_SOURCE = """
def validate_native(config, snapshot, expected):
    import torch
    from compare_attention.train import checkpoint, loop
    from compare_attention.train.identity import initialized_identity, parameter_fingerprint
    from compare_attention.models.gpt2 import GPT2
    from compare_attention.train.flops import estimate_step
    from compare_attention.train.precision import Precision
    from compare_attention.train.stopping import EarlyStopping
    payload = checkpoint.load_checkpoint(snapshot)
    for key in ('step', 'flops', 'cursor'):
        if type(payload[key]) is not int or payload[key] != expected[key]:
            raise ValueError('NATIVE_PIN_MISMATCH')
    if config.schedule.train_flops != expected['budget']:
        raise ValueError('NATIVE_BUDGET_MISMATCH')
    count = (expected['budget'] + expected['step_flops'] - 1) // expected['step_flops']
    if count != expected['target_updates'] or payload['flops'] >= expected['budget']:
        raise ValueError('NATIVE_EXHAUSTED_OR_TARGET_MISMATCH')
    pairs = tuple(loop._load(config.data, split) for split in ('train', 'val'))
    loop._validate_loaded_splits(config, *pairs)
    loop._seed(config.schedule.seed)
    model = GPT2(config.model)
    cost = estimate_step(config)
    if cost.total != expected['step_flops']:
        raise ValueError('NATIVE_ACCOUNTING_MISMATCH')
    measured = initialized_identity(config, model, torch.initial_seed())
    measured['accounting'] = cost.to_dict()
    device = loop._device()
    precision = None
    if config.schedule.dtype == 'float16':
        precision = (Precision(device, max_scale=config.schedule.amp_max_scale)
                     if config.schedule.amp_max_scale is not None else Precision(device))
    dtype = torch.float32 if precision is not None else config.schedule.torch_dtype()
    model.to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(loop._unique_parameters(model),
        lr=config.optim.learning_rate, weight_decay=config.optim.weight_decay)
    # The original validator checks runtime, initialization, config, aliases,
    # optimizer, scaler, RNG topology and integrity BEFORE any mutation.
    progress = checkpoint.restore_state(payload, model, optimizer, config, measured, precision)
    schedule = config.schedule
    stopper = EarlyStopping(schedule.early_stopping_patience,
                            schedule.early_stopping_min_delta, schedule.early_stopping_min_steps)
    stopper.load_state_dict(progress['early_stopping'])
    if 0 < stopper.patience <= stopper.bad_validations:
        raise ValueError('NATIVE_TERMINAL_STOP')
    history = progress['validation_history']
    wanted = list(range(schedule.val_every_steps, payload['step'] + 1, schedule.val_every_steps))
    if [row['flops'] for row in history] != [step * cost.total for step in wanted]:
        raise ValueError('NATIVE_INCOMPLETE_VALIDATION_HISTORY')
    replay = EarlyStopping(schedule.early_stopping_patience,
                          schedule.early_stopping_min_delta, schedule.early_stopping_min_steps)
    for step, row in zip(wanted, history, strict=True):
        if replay.observe(row['loss'], step):
            raise ValueError('NATIVE_TERMINAL_HISTORY')
    if replay.state_dict() != stopper.state_dict():
        raise ValueError('NATIVE_CONTROLLER_HISTORY_MISMATCH')
    # Read back all loaded state, not only a permissive load_state_dict return.
    restored = checkpoint.capture_state(model, optimizer, config,
        step=progress['step'], cursor=progress['cursor'], flops=progress['flops'],
        losses=progress['losses'], elapsed=progress['elapsed'],
        early_stopping=progress['early_stopping'], validation_history=history,
        identity=payload['identity'], precision=precision)
    if restored['integrity_sha256'] != payload['integrity_sha256']:
        raise ValueError('NATIVE_RESTORE_READBACK_MISMATCH')
    return {'step': progress['step'], 'flops': progress['flops'], 'cursor': progress['cursor'],
            'remaining_updates': count - progress['step'],
            'identity': checkpoint._identity(measured),
            'history_counts': {'losses': len(progress['losses']), 'validations': len(history)},
            'integrity_sha256': payload['integrity_sha256'],
            'restored_parameters_sha256': parameter_fingerprint(model),
            'optimizer_class': payload['optimizer_class'], 'precision': payload['precision'],
            'early_stopping': progress['early_stopping']}
"""


def encoded(value):
    """Canonical public metadata only."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True)
class RecoverySpec:
    """No caller overrides of checkpoint, scientific identity or original budget."""

    runtime: str
    run_name: str

    def __post_init__(self):
        label = r"[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"
        if (
            not re.fullmatch("runtimes/" + label, self.runtime)
            or not re.fullmatch(label, self.run_name)
            or self.run_name == ORIGIN
        ):
            raise ValueError("RECOVERY_IDENTITY")

    def value(self):
        """Return an independent exact public spec, never mutable shared state."""
        return {
            "schema_version": 1,
            "runtime": self.runtime,
            "run_name": self.run_name,
            "origin_run": ORIGIN,
            "origin_task": "36dabea9653c4544af3b0c1d4dfcbe5f",
            "revision": "f0d0efa4c8eb784677c79fed09b1ee6a27e04e2a",
            "snapshot_path": "runs/" + ORIGIN + "/checkpoints/step_00046500.pt",
            "snapshot_sha256": "ce288e4db2d47925c3876f93b9a874d344851010c9bc6bbd4acb7149602355bf",
            "bytes": 27788535,
            "step": 46500,
            "flops": 9838352400384000,
            "cursor": 88000,
            "step_flops": 211577470976,
            "budget": 10000000000000000,
            "target_updates": 47265,
            "archive_sha256": ARCHIVE_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "source_sha256": SOURCE_SHA,
            "data_fingerprint": DATA_SHA,
        }

    def sha256(self):
        """Bind exact source and destination without reading any file."""
        return hashlib.sha256(encoded(self.value())).hexdigest()


# Shared *generated* readback contracts; the host never imports target ML packages.
READBACK_SOURCE = """
import hashlib, json, os, re, sys
from pathlib import Path
sys.dont_write_bytecode = True

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def no_links(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('UNSAFE_RECOVERY_PATH')
    return path

def durable(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\\n') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())

def source_check(c):
    p, s = c['paths'], c['spec']
    for path in p.values():
        no_links(path)
    target = Path(p['target']).resolve()
    result = Path(p['result']).resolve()
    forbidden = [Path(p[key]).resolve() for key in ('package', 'venv', 'private')]
    for other in [result, *forbidden]:
        if target.is_relative_to(other) or other.is_relative_to(target):
            raise ValueError('OVERLAPPING_RECOVERY_PATH')
    for other in forbidden:
        if result.is_relative_to(other) or other.is_relative_to(result):
            raise ValueError('OVERLAPPING_RESULT_PATH')
    if result.exists() or result.is_symlink():
        raise ValueError('RESULT_EXISTS')
    if sha(p['archive']) != s['archive_sha256']:
        raise ValueError('RECOVERY_ARCHIVE_MISMATCH')
    if sha(Path(p['package']) / 'manifest.json') != s['manifest_sha256']:
        raise ValueError('RECOVERY_MANIFEST_MISMATCH')
    sys.path.insert(0, str(Path(p['package']) / 'tools'))
    from colab_package import verify_directory
    verify_directory(Path(p['package']))

def snapshot_check(c):
    p, s = c['paths'], c['spec']
    target = no_links(p['target'])
    snapshot = no_links(p['snapshot'])
    expected = {'step': s['step'], 'path': Path(s['snapshot_path']).name,
                'sha256': s['snapshot_sha256']}
    if {x.name for x in target.iterdir()} != {expected['path'], 'backup.json', 'latest.json'}:
        raise ValueError('SNAPSHOT_FILES_MISMATCH')
    latest = json.loads(no_links(target / 'latest.json').read_text())
    metadata = json.loads(no_links(target / 'backup.json').read_text())
    identity = {'schema_version': 1, 'run': s['origin_run'],
                'package_manifest_sha256': s['manifest_sha256'],
                'source_sha256': s['source_sha256'], 'data_fingerprint': s['data_fingerprint']}
    if (set(metadata) != {'identity', 'owned', 'nonce'} or metadata['identity'] != identity
            or not isinstance(metadata['nonce'], str)
            or not re.fullmatch('[a-f0-9]{32}', metadata['nonce'])
            or not isinstance(metadata['owned'], list) or not metadata['owned']):
        raise ValueError('SNAPSHOT_OWNERSHIP_MISMATCH')
    previous = -1
    for row in metadata['owned']:
        if (not isinstance(row, dict) or set(row) != {'step', 'path', 'sha256'}
                or type(row['step']) is not int or row['step'] <= previous
                or row['path'] != 'step_%08d.pt' % row['step']
                or not isinstance(row['sha256'], str)
                or not re.fullmatch('[a-f0-9]{64}', row['sha256'])):
            raise ValueError('SNAPSHOT_OWNED_ORDER_MISMATCH')
        previous = row['step']
    if latest != expected or metadata['owned'][-1] != expected:
        raise ValueError('SNAPSHOT_NOT_PINNED_LATEST')
    if snapshot.stat().st_size != s['bytes'] or sha(snapshot) != s['snapshot_sha256']:
        raise ValueError('SNAPSHOT_BYTES_MISMATCH')
    return {name: sha(target / name) for name in ('backup.json', 'latest.json', expected['path'])}
"""


def validate_status(value, binding):
    """Only bounded counters and public hashes may leave the recovery observer."""
    if (
        not isinstance(value, dict)
        or set(value)
        != {"spec_sha256", "inherited_updates", "new_updates", "proof_sha256", "helper_sha256"}
        or value["spec_sha256"] != binding["recovery_sha256"]
        or type(value["inherited_updates"]) is not int
        or value["inherited_updates"] != 46500
        or type(value["new_updates"]) is not int
        or not 0 <= value["new_updates"] <= 765
        or any(
            item is not None
            and (not isinstance(item, str) or not re.fullmatch("[a-f0-9]{64}", item))
            for item in (value["proof_sha256"], value["helper_sha256"])
        )
    ):
        raise ValueError("STATUS_INVALID")


def context(binding, **paths):
    """Bind the new isolated target paths to the immutable destination spec."""
    from pathlib import PurePosixPath  # pylint: disable=import-outside-toplevel

    spec = RecoverySpec(binding["runtime"], binding["run_name"])
    if binding.get("recovery_sha256") != spec.sha256() or binding["sha256"] != ARCHIVE_SHA:
        raise ValueError("RECOVERY_BINDING_MISMATCH")
    root = PurePosixPath(paths["root"])
    paths = dict(
        paths,
        target=str(root / "recovery-input"),
        snapshot=str(root / "recovery-input/step_00046500.pt"),
    )
    return {
        "binding": dict(binding),
        "spec": spec.value(),
        "paths": paths,
        "spec_sha256": spec.sha256(),
    }


def native_program(c):
    """Frozen config parser and exact validator, in a credential-free child."""
    return (
        NATIVE_SOURCE
        + "\n"
        + (
            "import json, os, sys\nfrom pathlib import Path\n"
            "from compare_attention.cli.train import load_experiment\n"
            "from compare_attention.train.config import TrainConfig\n" + "c = " + repr(c) + "\n"
            "p = c['paths']\n"
            "sys.path.insert(0, str(Path(p['package']) / 'tools'))\n"
            "from colab_package import CONFIGS\n"
            "args = load_experiment(Path(p['package']) / 'release' / CONFIGS['deltanet'])\n"
            "excluded = {'config','kv','result_dir','resume_from','tracker','hf_repo'}\n"
            "raw = args.model_dump(exclude=excluded)\n"
            "raw['data'] = Path(p['package']) / 'data'\n"
            "config = TrainConfig.model_validate(raw)\n"
            "measured = validate_native(config, Path(p['snapshot']), c['spec'])\n"
            "with (Path(p['root']) / 'restore/semantic.json').open('x') as stream:\n"
            "    json.dump(measured, stream, sort_keys=True, allow_nan=False)\n"
            "    stream.flush()\n    os.fsync(stream.fileno())\n"
        )
    )


def restore_source(c):
    """Exact separately delivered helper; never add anything to frozen package."""
    return (
        READBACK_SOURCE
        + "\nc = "
        + repr(c)
        + "\n"
        + (
            "source_check(c)\n"
            "from colab_runner import preflight, clean_environment, venv_python\n"
            "preflight()\n"
            "import subprocess\n"
            "p, s = c['paths'], c['spec']\n"
            "target = Path(p['target'])\n"
            "if target.exists() or target.is_symlink():\n"
            "    raise ValueError('RESTORE_TARGET_EXISTS')\n"
            "python = str(venv_python(Path(p['venv'])))\n"
            "command = [python, str(Path(p['package']) / 'tools/colab_checkpoint_backup.py'),\n"
            "           'restore', '--package-dir', p['package'], '--env-file', p['private'],\n"
            "           '--run', s['origin_run'], '--revision', s['revision'],\n"
            "           '--target', p['target']]\n"
            "subprocess.run(command, env=clean_environment(), check=True,\n"
            "               stdin=subprocess.DEVNULL,\n"
            "               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "files = snapshot_check(c)\n"
            "subprocess.run([python, '-B', '-c', " + repr(native_program(c)) + "],\n"
            "               env=clean_environment(), check=True, stdin=subprocess.DEVNULL,\n"
            "               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "if snapshot_check(c) != files:\n    raise ValueError('SNAPSHOT_CHANGED')\n"
            "semantic = Path(p['root']) / 'restore/semantic.json'\n"
            "proof = {'schema_version': 1, 'verified': True, 'binding': c['binding'],\n"
            "         'spec': s, 'spec_sha256': c['spec_sha256'], 'paths': p, 'files': files,\n"
            "         'semantic_sha256': sha(semantic), 'helper_sha256': sha(__file__)}\n"
            "durable(Path(p['root']) / 'restore/proof.json', proof)\n"
        )
    )


def resume_guard(c, helper_sha, proof_sha):
    """No transport: require exact reviewed proof and rehash every source byte."""
    if not all(
        isinstance(s, str) and re.fullmatch("[a-f0-9]{64}", s) for s in (helper_sha, proof_sha)
    ):
        raise ValueError("RESTORE_PROOF_SHA_REQUIRED")
    return (
        READBACK_SOURCE
        + "\nc = "
        + repr(c)
        + "\n"
        + (
            "source_check(c)\n"
            "p = c['paths']\n"
            "root = Path(p['root'])\n"
            "proof_path = no_links(root / 'restore/proof.json')\n"
            "if sha(proof_path) != "
            + repr(proof_sha)
            + ":\n    raise ValueError('RESTORE_PROOF_CHANGED')\n"
            "proof = json.loads(proof_path.read_text())\n"
            "helper = no_links(root / 'restore/recovery.py')\n"
            "semantic = no_links(root / 'restore/semantic.json')\n"
            "expected = {'schema_version': 1, 'verified': True, 'binding': c['binding'],\n"
            "    'spec': c['spec'], 'spec_sha256': c['spec_sha256'], 'paths': p,\n"
            "    'files': snapshot_check(c), 'semantic_sha256': sha(semantic),\n"
            "    'helper_sha256': " + repr(helper_sha) + "}\n"
            "if proof != expected or sha(helper) != expected['helper_sha256']:\n"
            "    raise ValueError('RESTORE_PROOF_MISMATCH')\n"
        )
    )
