"""Fixed remote programs. Never interpolate credentials or arbitrary user code."""

from colab_runner import PREFLIGHT

MARKER = "COMPARE_ATTENTION_JOB_V1:"
ROOT = "/content/compare-attention-jobs/"
PACKAGE = "/content/compare-attention-package"
VENV = "/content/compare-attention-venv"
PRIVATE = "/content/private/kaggle-env.env"
ARCHIVE = "/content/colab-package.zip"
RESULTS = "/content/compare-attention-results/"


def emit(expression):
    """Single controlled result marker, never arbitrary stdout."""
    return "print(" + repr(MARKER) + " + json.dumps(" + expression + "))\n"


def marker_code(binding, nonce):
    """Hardware check FIRST, reserve owned directory before private delivery."""
    root = ROOT + binding["run_name"]
    return (
        PREFLIGHT
        + """import json, os
from pathlib import Path
binding = """
        + repr(binding)
        + """
root = Path("""
        + repr(root)
        + """)
root.mkdir(parents=True, exist_ok=False)
(root / 'identity.json').write_text(json.dumps(binding))
private = Path('/content/private')
private.mkdir(mode=0o700, exist_ok=False)
name = """
        + repr("compare-transport-" + nonce)
        + """
with (Path('/content') / name).open('x') as output:
    output.write("""
        + repr(nonce)
        + """)
"""
        + emit("{'guard_passed': True}")
    )


def finalize_code(binding, stage):
    """Rehash before exposing archive; chmod private input before linking."""
    root = ROOT + binding["run_name"]
    return (
        """import hashlib, json, os, stat
from pathlib import Path
binding = """
        + repr(binding)
        + """
root = Path("""
        + repr(root)
        + """)
if json.loads((root / 'identity.json').read_text()) != binding:
    raise RuntimeError('IDENTITY_MISMATCH')
archive = Path("""
        + repr("/content/" + stage)
        + """)
if archive.is_symlink() or not archive.is_file():
    raise RuntimeError('ARCHIVE_INVALID')
with archive.open('rb') as stream:
    if hashlib.file_digest(stream, 'sha256').hexdigest() != binding['sha256']:
        raise RuntimeError('ARCHIVE_SHA_MISMATCH')
private = Path('/content/private/input.env')
if private.is_symlink() or not private.is_file():
    raise RuntimeError('PRIVATE_INVALID')
private.chmod(0o600)
os.link(archive, """
        + repr(ARCHIVE)
        + """)
os.link(private, """
        + repr(PRIVATE)
        + """)
with (root / 'inputs.json').open('x') as output:
    json.dump({'verified': True, 'sha256': binding['sha256']}, output)
"""
        + emit("{'inputs_verified': True, 'sha256': binding['sha256']}")
    )


def supervisor_code(command, state_path, log_path, bootstrap, binding):
    """One detached supervisor waits for one actual runner and atomically records exit."""
    return (
        """import json, os, subprocess, sys
from pathlib import Path
state_path = Path("""
        + repr(state_path)
        + """)
binding = """
        + repr(binding)
        + """
def save(state):
    temporary = state_path.with_suffix('.tmp')
    with temporary.open('w') as output:
        json.dump(state, output)
        output.flush()
        os.fsync(output.fileno())
    # Only atomic LOCAL publication is retried; never a spawn or network operation.
    import time
    for attempt in range(20):
        try:
            os.replace(temporary, state_path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)
state = dict(binding, state='running', supervisor_pid=os.getpid())
save(state)
try:
    with open("""
        + repr(log_path)
        + """, 'ab', buffering=0) as log:
        bootstrap = """
        + repr(bootstrap)
        + """
        if bootstrap:
            prepared = subprocess.run([sys.executable, '-B', '-c', bootstrap],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, check=False)
            if prepared.returncode:
                state.update(state='completed', returncode=prepared.returncode,
                             failure='BOOTSTRAP_FAILED')
                save(state)
                raise SystemExit(0)
        command = """
        + repr(command)
        + """
        if command[0] == '__KERNEL_PYTHON__':
            command[0] = sys.executable
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL,
            stdout=log, stderr=log)
        state['child_pid'] = child.pid
        save(state)
        state.update(state='completed', returncode=child.wait())
except Exception:
    state.update(state='completed', returncode=125, failure='SUPERVISOR_FAILED')
save(state)
"""
    )


def launch_code(binding, phase, bootstrap="", *, helper_sha=None, proof_sha=None):
    """Exclusive remote stage dir and pending state precede the only spawn."""
    if phase not in ("setup", "train", "restore", "resume"):
        raise ValueError("INVALID_PHASE")
    if phase == "train" and "recovery_sha256" in binding:
        raise ValueError("RECOVERY_REQUIRES_EXPLICIT_RESUME")
    root = ROOT + binding["run_name"]
    helper, guard = "", ""
    stage_name = "train" if phase == "resume" else phase
    if phase in ("restore", "resume"):
        from colab_api_recovery import context, restore_source, resume_guard  # pylint: disable=import-outside-toplevel

        recovery = context(
            binding,
            root=root,
            package=PACKAGE,
            venv=VENV,
            private=PRIVATE,
            archive=ARCHIVE,
            result=RESULTS + binding["run_name"],
        )
        if phase == "restore":
            helper = restore_source(recovery)
        else:
            guard = resume_guard(recovery, helper_sha, proof_sha)
            bootstrap = PREFLIGHT + guard
    command = [
        "__KERNEL_PYTHON__",
        "-B",
        PACKAGE + "/tools/colab_runner.py",
        "--package-dir",
        PACKAGE,
        "--venv",
        VENV,
        "--storage",
        "hf",
    ]
    command += (
        ["--setup"]
        if phase == "setup"
        else [
            "--run",
            "--family",
            "deltanet",
            "--result-dir",
            RESULTS + binding["run_name"],
            "--env-file",
            PRIVATE,
        ]
    )
    if phase == "restore":
        command = ["__KERNEL_PYTHON__", "-B", root + "/restore/recovery.py"]
    if phase == "resume":
        command += ["--resume-from", root + "/recovery-input/step_00046500.pt"]
    code = supervisor_code(
        command,
        root + "/" + stage_name + "/state.json",
        root + "/" + stage_name + "/output.log",
        bootstrap,
        dict(binding, phase=phase),
    )
    return (
        """import json, os, subprocess, sys
from pathlib import Path
binding = """
        + repr(binding)
        + """
root = Path("""
        + repr(root)
        + """)
if json.loads((root / 'identity.json').read_text()) != binding:
    raise RuntimeError('IDENTITY_MISMATCH')
inputs = json.loads((root / 'inputs.json').read_text())
if inputs != {'verified': True, 'sha256': binding['sha256']}:
    raise RuntimeError('INPUTS_UNVERIFIED')
phase = """
        + repr(phase)
        + """
if phase in ('train', 'restore', 'resume'):
    setup = json.loads((root / 'setup/state.json').read_text())
    if any(setup.get(k) != v for k, v in binding.items()):
        raise RuntimeError('SETUP_IDENTITY_MISMATCH')
    if setup.get('state') != 'completed' or setup.get('returncode') != 0:
        raise RuntimeError('SETUP_NOT_VERIFIED')
    result = Path("""
        + repr(RESULTS + binding["run_name"])
        + """)
    if result.exists() or result.is_symlink():
        raise RuntimeError('RESULT_EXISTS')
if phase == 'resume':
    restored = json.loads((root / 'restore/state.json').read_text())
    if any(restored.get(k) != v for k, v in binding.items()):
        raise RuntimeError('RESTORE_IDENTITY_MISMATCH')
    if restored.get('state') != 'completed' or restored.get('returncode') != 0:
        raise RuntimeError('RESTORE_NOT_VERIFIED')
"""
        + guard
        + "\nstage = root / "
        + repr(stage_name)
        + "\n"
        + """
stage.mkdir(exist_ok=False)
helper = """
        + repr(helper)
        + """
if helper:
    with (stage / 'recovery.py').open('x', encoding='utf-8', newline='\\n') as output:
        output.write(helper)
        output.flush()
        os.fsync(output.fileno())
with (stage / 'state.json').open('x') as output:
    json.dump(dict(binding, phase=phase, state='spawn_unknown'), output)
    output.flush()
    os.fsync(output.fileno())
script = stage / 'supervisor.py'
with script.open('x') as output:
    output.write("""
        + repr(code)
        + """)
with (stage / 'supervisor.log').open('ab', buffering=0) as sink:
    child = subprocess.Popen([sys.executable, '-B', str(script)],
        stdin=subprocess.DEVNULL, stdout=sink, stderr=sink, start_new_session=True)
"""
        + emit("{'phase': phase, 'state': 'spawned', 'pid': child.pid}")
    )


def status_code(binding):
    """Read only owned state/metrics, returning a bounded numeric projection."""
    return (
        """import json, math, os
from pathlib import Path
binding = """
        + repr(binding)
        + """
root = Path("""
        + repr(ROOT + binding["run_name"])
        + """)
if json.loads((root / 'identity.json').read_text()) != binding:
    raise RuntimeError('IDENTITY_MISMATCH')
answer = {'sha256': binding['sha256'], 'stages': {}, 'metrics': {}, 'hf_checkpoint': None}
phases = ('setup', 'restore', 'train') if 'recovery_sha256' in binding else ('setup', 'train')
for phase in phases:
    path = root / phase / 'state.json'
    if not path.exists():
        answer['stages'][phase] = {'state': 'not_started'}
        continue
    state = json.loads(path.read_text())
    if any(state.get(k) != v for k, v in binding.items()):
        raise RuntimeError('STATE_IDENTITY_MISMATCH')
    if state.get('state') not in ('spawn_unknown', 'running', 'completed'):
        raise RuntimeError('STATE_INVALID')
    projected = {'state': state['state']}
    if type(state.get('returncode')) is int:
        projected['returncode'] = state['returncode']
    if state['state'] == 'running' and type(state.get('supervisor_pid')) is int:
        try:
            os.kill(state['supervisor_pid'], 0)
            projected['process_present'] = True
        except ProcessLookupError:
            projected['process_present'] = False
    answer['stages'][phase] = projected
metrics = Path("""
        + repr(RESULTS + binding["run_name"] + "/metrics.jsonl")
        + """)
if metrics.is_file():
    with metrics.open('rb') as stream:
        stream.seek(max(0, metrics.stat().st_size - 65536))
        lines = stream.read(65536).splitlines()
    for line in lines:
        try:
            value = json.loads(line)
        except (ValueError, UnicodeError):
            continue
        for key in ('step', 'steps', 'update', 'train_loss', 'val_loss',
                    'loss', 'train_flops', 'flops'):
            number = value.get(key) if isinstance(value, dict) else None
            if type(number) in (int, float) and math.isfinite(number):
                answer['metrics'][key] = number
receipt_path = metrics.parent.with_name(metrics.parent.name + '.hf-checkpoint-receipt.json')
if receipt_path.is_file() and receipt_path.stat().st_size < 65536:
    import re
    receipt = json.loads(receipt_path.read_text())
    identity = receipt.get('identity', {})
    revision = receipt.get('revision')
    expected_source = '1beba22ca8f9897e0b7367e07f4a016559946f1183fc8413deff2490c6e85ef1'
    expected_data = '1c660aa5f54d88ea9e0c1ce4eba0c25084eee164c55ab6c1d80d5db14d3e59fb'
    if (receipt.get('verified') is True and identity.get('run') == binding['run_name']
            and identity.get('source_sha256') == expected_source
            and identity.get('data_fingerprint') == expected_data
            and isinstance(revision, str) and re.fullmatch('[a-f0-9]{40}', revision)):
        answer['hf_checkpoint'] = {'verified': True, 'revision': revision}
        if type(receipt.get('step')) is int and receipt['step'] >= 0:
            answer['hf_checkpoint']['step'] = receipt['step']
"""
        + (
            """
answer['recovery'] = {'spec_sha256': binding['recovery_sha256'], 'inherited_updates': 46500,
                      'new_updates': 0, 'proof_sha256': None, 'helper_sha256': None}
observed = answer['metrics'].get('flops', 0)
if type(observed) is int and observed % 211577470976 == 0:
    answer['recovery']['new_updates'] = max(0, observed // 211577470976 - 46500)
restored = answer['stages']['restore']
if restored.get('state') == 'completed' and restored.get('returncode') == 0:
    import hashlib
    proof_path = root / 'restore/proof.json'
    helper_path = root / 'restore/recovery.py'
    proof = json.loads(proof_path.read_text())
    if proof.get('binding') != binding or proof.get('verified') is not True:
        raise RuntimeError('RESTORE_PROOF_MISMATCH')
    answer['recovery']['proof_sha256'] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    answer['recovery']['helper_sha256'] = hashlib.sha256(helper_path.read_bytes()).hexdigest()
"""
            if "recovery_sha256" in binding
            else ""
        )
        + emit("answer")
    )
