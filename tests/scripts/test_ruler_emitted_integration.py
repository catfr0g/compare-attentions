"""Actual public NORMAL RULER emission, without SDK or provider access."""

import json
from pathlib import Path


def public_inputs(tmp_path):
    from normal_ruler_scope import AUTHORIZATION_SHA256
    from unified_normal_profiles import RULER, selected

    root = Path("E:/HermesArtifacts/compare-attention/ruler-variable-tracking-three-seeds-v1")
    base = Path("E:/HermesArtifacts/compare-attention/tiny-shakespeare-three-seeds-v1")
    previous = json.loads(
        (base / "provider04/cases02/TinyClassic202/inputs-unbound.json").read_bytes()
    )
    source = root / "implementation05/source03"
    generation = tmp_path / "generation.json"
    generation.write_text(
        json.dumps(
            {
                "schema": "UNIFIED_NORMAL_GENERATION_INTENT_V1",
                "nonce": "a6ab17b20eb1491d854e0c0e2c1ee8b1",
                "case_id": "RulerClassic202",
                "source_identity_sha256": selected(RULER)["source"],
                "authorization": {
                    "path": str(root / "USER-AUTHORIZATION01.json"),
                    "sha256": AUTHORIZATION_SHA256,
                },
            }
        )
    )
    return {
        "sorting_candidate": previous["sorting_candidate"],
        "runtime_record": previous["runtime_record"],
        "source_profile": RULER,
        "normal_package": str(source / "normal-package"),
        "normal_case": str(source / "derived-case.json"),
        "normal_plan": str(source / "normal-plan.json"),
        "normal_recipe": str(source / "recipe.yaml"),
        "predecessors": [],
        "generation": str(generation),
    }, source / "ruler-normal-source.zip"


def test_real_operator_builds_independent_unbound_ruler(tmp_path):
    from unified_normal_contract import descriptor
    from unified_normal_operator import build

    inputs, archive = public_inputs(tmp_path)
    candidate = build(inputs, archive, tmp_path / "candidate")
    _, request = descriptor(candidate)
    assert request["run"]["id"] == "RulerClassic202"
    assert request["predecessors"] == []
    assert request["run"]["step_flops"] == 45826965504
    assert request["run"]["updates"] == 109107
    authority = json.loads((candidate / "payload/unified/normal-authority.json").read_bytes())
    assert authority["status"] == "UNBOUND_REVIEW_AND_DECISION_REQUIRED"
    for seam in (candidate / "request-build/source", candidate / "payload/source/src"):
        assert (seam / "normal_ruler_scope.py").is_file()
        assert (seam / "normal_training_tracking.py").is_file()
        assert (seam / "unified_normal_ruler.py").is_file()
    worker = candidate / "worker-build"
    assert (worker / "data/ruler/meta.json").is_file()
    assert not (worker / "data/tiny").exists()
    exercise_public(candidate, tmp_path / "public")
    exercise_kernel(candidate, tmp_path / "kernel")


def exercise_public(candidate, output):
    """Fresh isolated interpreter, only emitted first-party modules."""
    import subprocess
    import sys

    output.mkdir()
    code = """import sys,json
from pathlib import Path
root,output=map(Path,sys.argv[1:]);host=root/'payload/source/src'
sys.path.insert(0,str(host))
from seed202_pilot import cli,preparation
from unified_normal_contract import descriptor
from unified_normal_authority import binding
assert Path(cli.__file__).is_relative_to(host)
assert binding(root)['normal_authority_bound'] is False
base=['--candidate-dir',str(root),'--case','RulerClassic202']
assert cli.main(['check',*base])==0
assert cli.main(['next',*base])==0
plan=output/'singleton.json'
assert cli.main(['make-plan',*base,'--journal',str(output/'journals'),
                 '--output',str(plan)])==0
subject=json.loads(plan.read_bytes())
assert subject['source_authority']['fresh']['normal_authority_bound'] is False
from seed202_pilot.seal import _plan
_plan(subject)
_,request=descriptor(root)
data=Path(request['run']['config']['data'])
for action in ('publish','submit'):
    argv=['prepare',*base,'--action',action,'--task-data',str(data),
          '--plan',str(plan),'--review',str(output/'MISSING-REVIEW.json'),
          '--journal',str(output/(action+'-journal.json')),
          '--output',str(output/(action+'-preparation.json'))]
    args=cli._parser().parse_args(argv)
    paths=set(preparation.paths(args))
    assert {str(p.resolve()) for p in data.iterdir()} <= {str(Path(p).resolve()) for p in paths}
    assert args.mqar_data is None
    from unified_normal_authority import before_auth
    try:
        before_auth(args)
    except ValueError as error:
        assert error.args == ('UNIFIED_NORMAL_EXECUTION_AUTHORITY_REQUIRED',)
    else:
        raise AssertionError('UNBOUND authority admitted')
    cli._provider=lambda *_: (_ for _ in ()).throw(AssertionError('SDK entry'))
    assert cli.main(argv)==1
    for command in ('probe','seal',action):
        actual=[command,*argv[1:]]
        parsed=cli._parser().parse_args(actual)
        assert parsed.task_data==data and parsed.mqar_data is None
        assert set(preparation.paths(parsed))==paths
        assert cli.main(actual)==1
assert not (output/'publish-preparation.json').exists()
print('EMITTED_PUBLIC_UNBOUND_OK')
"""
    child = subprocess.run(
        [
            "uv",
            "run",
            "--offline",
            "--no-project",
            "--no-sync",
            "--python",
            sys.executable,
            "python",
            "-I",
            "-B",
            "-c",
            code,
            str(candidate),
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    assert child.returncode == 0, child.stderr + child.stdout
    assert "EMITTED_PUBLIC_UNBOUND_OK" in child.stdout


def exercise_kernel(candidate, tmp_path):
    import runpy
    import sys
    import time

    import pytest

    namespace = runpy.run_path(
        str(candidate / "payload/deploy/RulerClassic202/kernel/kernel.py"),
        run_name="ruler_emitted_no_main",
    )
    main = namespace["normal_main"]
    scope = main.__globals__
    scope["inspect_hardware"] = lambda _driver: None  # NOT CUDA qualification
    scope["driver_directory"] = lambda: tmp_path
    archives = {
        "seed202-pilot.zip.bin": candidate / "payload/worker/seed202-pilot.zip.bin",
        "scientific.zip.bin": candidate / "payload/inherited/sorting-science.zip.bin",
    }
    scope["resolve_mount"] = lambda _root, _owner, _slug, name: archives[name]
    observed = {}

    def trace(frame, event, _arg):
        if frame.f_code is main.__code__ and event == "exception":
            observed.update(frame.f_locals)
        return trace

    previous = sys.gettrace()
    sys.settrace(trace)
    try:
        with pytest.raises(FileNotFoundError) as failure:
            main(tmp_path / "mounts", tmp_path / "working", tmp_path / "packages", time.monotonic())
    finally:
        sys.settrace(previous)
    assert Path(failure.value.filename).name == "independent-review.json"
    assert observed["data"] == observed["release"] / "data/ruler"
    assert len(list(observed["data"].iterdir())) == 7
    assert "setup" not in observed
