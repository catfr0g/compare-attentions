"""Render the inherited working two-stage hardware/setup/UV route for this tail."""
# ruff: noqa: E501 -- saved source literals retain inherited guard text

import ast
from pathlib import Path

from mqar_final_tail_native import digest, require

INHERITED = "3675f6a9f914194bbcfdeeea60d6994480205c1d22d88615429825eb48d7ac8e"
SCIENCE = "71872615b0cfbe2406cd08762cc02c390c88a70b6bb5f4bb38d6571830c8a3bb"
BODY = """
def main(mount_root=Path('/kaggle/input'), working=Path('/kaggle/working'), package_root=Path('/kaggle/temp')):
    started = time.monotonic()
    driver = driver_directory()
    inspect_hardware(driver)
    operator = resolve_mount(mount_root, 'zimike', BINDING['slug'], 'recovery-operator.zip.bin')
    release = extract_archive(operator, package_root / ('mqar202-final10780-tail-' + BINDING['nonce']), BINDING['operator_sha256'])
    sys.path.insert(0, str(release / 'src'))
    from mqar_final_tail_package import check
    check(release)
    source = resolve_mount(mount_root, 'zimike', 'sorting-quality-input-fcff1b5daf574b70', 'scientific.zip.bin')
    science = extract_archive(source, package_root / ('mqar202-final10780-tail-science-' + BINDING['nonce']), BINDING['science_sha256'])
    check(science)
    compact = resolve_mount(mount_root, 'zimike', BINDING['slug'], 'recovery-input.zip.bin')
    original = extract_archive(compact, package_root / ('mqar202-final10780-tail-input-' + BINDING['nonce']), BINDING['input_sha256'])
    check(original)
    decision = resolve_mount(mount_root, 'zimike', BINDING['slug'], 'execution-decision.json')
    review = resolve_mount(mount_root, 'zimike', BINDING['slug'], 'independent-review.json')
    data = resolve_mount(mount_root, 'zimike', 'ca-kv-1c660aa5f54d88ea9e0c1ce4', 'meta.json').parent
    private = resolve_mount(mount_root, 'zimike', 'k64-private-20261109a1b2', 'private.env')
    sys.path[:0] = [str(science / 'operator/inherited'), str(science / 'tools')]
    from session import setup
    from process_control import bounded_run
    work = working / ('mqar202-final10780-tail-' + BINDING['nonce'])
    uv, python, env, _topology = setup(science, work, started + 1800)
    env.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='0', CLEARML_OFFLINE_MODE='0', DDP_BENCH_UV=str(uv))
    env['PYTHONPATH'] = os.pathsep.join([str(release / 'src')])
    env['C64_RECOVERY_RUNTIME_DIAGNOSTIC'] = str(work / 'runtime-qualification.json')
    command = [str(uv), 'run', '--offline', '--no-project', '--no-sync', '--python', str(python), 'python', '-B',
               str(release / 'src/mqar_final_tail_cli.py'), '--release', str(release), '--science', str(science),
               '--input', str(original), '--work', str(work), '--private', str(private), '--decision', str(decision), '--data', str(data), '--nonce', BINDING['nonce']]
    bounded_run(command, env, science, 7200 - (time.monotonic() - started))

if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            main()
    except Exception as error:
        print(json.dumps(public_error(error, 'entry'), sort_keys=True))
        raise SystemExit(1) from None
"""


def render(nonce, operator_sha256, input_sha256, *, guard_path=None):
    import re

    require(type(nonce) is str and re.fullmatch("[0-9a-f]{32}", nonce), "FRESH_NONCE")
    require(
        nonce not in {"32711a45ea089d55f5b34eb741f89e01", "814d3ba65319c419f6206aac1a528047"},
        "CONSUMED_TRAIN_NONCE",
    )
    path = (
        Path(guard_path)
        if guard_path is not None
        else Path(__file__).with_name("inherited_kernel.py")
    )
    require(digest(path) == INHERITED, "INHERITED_GUARD_BYTES")
    text = path.read_text(encoding="utf-8")
    names = {
        "driver_directory",
        "resolve_mount",
        "hardware",
        "inspect_hardware",
        "extract_archive",
        "public_error",
        "CLASSES",
        "CATEGORIES",
        "FILES",
        "FUNCTIONS",
    }
    chosen = []
    for node in ast.parse(text).body:
        name = (
            node.name
            if isinstance(node, ast.FunctionDef)
            else node.targets[0].id
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            else None
        )
        if name in names:
            chosen.append(ast.get_source_segment(text, node))
    require(len(chosen) == len(names), "GUARD_FUNCTION_SET")
    binding = {
        "nonce": nonce,
        "slug": "mqar202-final10780-tail-input-" + nonce[:16],
        "operator_sha256": operator_sha256,
        "input_sha256": input_sha256,
        "science_sha256": SCIENCE,
    }
    prefix = '# fmt: off\n# ruff: noqa: E402, I001, E501\nimport os\nimport sys\nsys.dont_write_bytecode=True\nos.environ["PYTHONDONTWRITEBYTECODE"]="1"\nimport contextlib\nimport hashlib\nimport json\nimport logging\nimport platform\nimport stat\nimport subprocess\nimport time\nimport zipfile\nfrom pathlib import Path, PurePosixPath\n'
    source = prefix + "\nBINDING = " + repr(binding) + "\n\n" + "\n\n".join(chosen) + BODY
    compile(source, "kernel.py", "exec")
    metadata = {
        "id": "zimike/mqar202-final10780-tail-" + nonce[:16],
        "title": "mqar202-final10780-tail-" + nonce[:16],
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [
            "zimike/" + binding["slug"] + "/1",
            "zimike/sorting-quality-input-fcff1b5daf574b70/1",
            "zimike/k64-private-20261109a1b2/1",
            "zimike/ca-kv-1c660aa5f54d88ea9e0c1ce4/1",
        ],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    return source, metadata
