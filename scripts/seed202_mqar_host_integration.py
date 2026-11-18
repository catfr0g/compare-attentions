"""Exact source integration of typed MQAR continuation host consumers."""

from pathlib import Path

MODE = "MQAR_CLASSIC_STEP980_V1"
NONCE = "4a2060304e6b47229b64b0647fc138d1"
BASELINE_SHA = "581406e1f8569fcfa50d319033515209962a0b4c4f67d98947905cd226aa1712"
ORIGINAL_SOURCE_SHA = "19ff980d566691e4fde21130f3fff90400e8773011206b86a09cb21e5661ff6a"


def integrate(package):
    """Specialize a new adapter; preserve original guarded host_resume.py bytes."""
    from build_seed202_mqar_continuation import replace_once

    package = Path(package)
    raw = (package / "host_resume.py").read_text(encoding="utf-8")
    mqar_scope = (
        'deployment["case_id"] == "mqar-k64-classic-seed202"\n'
        '        and deployment.get("mode") == "' + MODE + '"'
    )
    substitutions = {
        "d0997ffbc8ddf0a2fbc69f94f67489049620729c40beba9e714e083021caca5f": BASELINE_SHA,
        "89e88cdb777b0af32b53807709a027b6454ebdd8bb3621b984ae31bafe1fb41b": ORIGINAL_SOURCE_SHA,
        "zimike/seed202-pilot-814d3ba65319c419": "zimike/seed202-pilot-786acf2c5ec864ce",
        "inherited/seed202-candidate03-original.zip.bin": "inherited/mqar-original.zip.bin",
        "deploy/sorting-classic-seed202/": "deploy/mqar-k64-classic-seed202/",
        "145376087": "145778849",
        "32711a45ea089d55f5b34eb741f89e01": NONCE,
        "32711a45ea089d55": NONCE[:16],
        'deployment["case_id"] == "sorting-classic-seed202"': mqar_scope,
    }
    for before, after in substitutions.items():
        if before not in raw:
            raise ValueError("EXACT_HOST_ADAPTER_SEAM")
        raw = raw.replace(before, after)
    raw = replace_once(
        raw,
        "    candidate = Path(candidate)\n    verify_extracted(candidate)",
        "    scoped_targets(deployment)\n"
        "    candidate = Path(candidate)\n    verify_extracted(candidate)",
    )
    raw = replace_once(
        raw,
        '    resume = pin((deploy / "resume-input/resume.zip.bin").read_bytes())',
        '    resume = pin((deploy / "resume-input/resume.zip.bin").read_bytes())\n'
        "    from seed202_pilot.mqar_publisher import local_bundle\n"
        '    resume["files"] = local_bundle(deploy, deployment)',
    )
    raw = replace_once(
        raw,
        '        "original_worker": {\n            "target": old_meta["dataset_sources"][0],',
        '        "original_worker": {\n            "numeric_id": 13454634,\n'
        '            "target": old_meta["dataset_sources"][0],',
    )
    (package / "mqar_host_resume.py").write_text(raw, encoding="utf-8", newline="\n")
    router = Path(__file__).with_name("seed202_mqar_host_router.py").read_text(encoding="utf-8")
    (package / "resume_router.py").write_text(router, encoding="utf-8", newline="\n")
    for name in ("fresh_mode", "host_operator", "preparation", "preflight"):
        path = package / (name + ".py")
        text = path.read_text(encoding="utf-8")
        if "from seed202_pilot.host_resume import " not in text:
            raise ValueError("EXACT_HOST_CONSUMER_IMPORT")
        text = text.replace(
            "from seed202_pilot.host_resume import ", "from seed202_pilot.resume_router import "
        )
        if name == "fresh_mode":
            text = replace_once(
                text,
                '    return deployment.get("mode") == MODE or "resume_input_target" in deployment',
                '    if "resume_input_target" in deployment:\n'
                "        from seed202_pilot.resume_router import scoped_targets\n\n"
                "        scoped_targets(deployment)\n"
                '    return deployment.get("mode") == MODE or "resume_input_target" in deployment',
            )
        path.write_text(text, encoding="utf-8", newline="\n")
    path = package / "seal.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '"seed202_pilot.host_resume", fromlist=["binding"]',
        '"seed202_pilot.resume_router", fromlist=["binding"]',
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    path = package / "host_operator.py"
    text = path.read_text(encoding="utf-8")
    text = replace_once(
        text,
        "    from seed202_pilot.provider import dataset_readback, kernel_readback",
        "    from seed202_pilot.provider import kernel_readback",
    )
    text = text.replace(
        "return dataset_readback(provider, target, filename, payload)",
        'return __import__("seed202_pilot.mqar_publisher", fromlist=["readback"]).readback(\n'
        "                provider, target, filename, payload, deploy, deployment)",
    )
    text = replace_once(
        text,
        '            "sha256": digest(folder / filename),\n        }',
        '            "sha256": digest(folder / filename),\n'
        '            **({"files": __import__("seed202_pilot.mqar_publisher",\n'
        '                fromlist=["local_bundle"]).local_bundle(deploy, deployment)}\n'
        '               if resume and deployment.get("mode") == '
        '"MQAR_CLASSIC_STEP980_V1" else {}),\n'
        "        }",
    )
    text = replace_once(
        text,
        '                evidence["resume_remote"] = dataset_readback(',
        '                evidence["resume_remote"] = __import__(\n'
        '                    "seed202_pilot.mqar_publisher", fromlist=["readback"]).readback(',
    )
    text = replace_once(
        text,
        '                    (deploy / "resume-input/resume.zip.bin").read_bytes(),\n'
        "                )",
        '                    (deploy / "resume-input/resume.zip.bin").read_bytes(),\n'
        "                    deploy, deployment,\n                )",
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    path = package / "preflight.py"
    text = replace_once(
        path.read_text(encoding="utf-8"),
        '        if method == "SaveKernel":',
        '        if self.action == "publish-resume" and deployment.get("mode") == '
        '"MQAR_CLASSIC_STEP980_V1":\n'
        "            from seed202_pilot.mqar_publisher import transport\n"
        "            return transport(self, method, request)\n"
        '        if method == "SaveKernel":',
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    path = package / "cli.py"
    text = replace_once(
        path.read_text(encoding="utf-8"),
        "            from seed202_pilot.host_actions import command_scope",
        "            from seed202_pilot.mqar_command_scope import command_scope",
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    path = package / "preparation.py"
    text = replace_once(
        path.read_text(encoding="utf-8"),
        "    from seed202_pilot.qualified_bridge import evidence_paths\n",
        '    rows = json.loads((candidate / "PLAN-UNAPPROVED.json").read_bytes())["deployments"]\n'
        '    if any(row["case_id"] == args.case and row.get("mode")\n'
        '           == "MQAR_CLASSIC_STEP980_V1" for row in rows):\n'
        "        require(args.plan is not None and args.review is not None,\n"
        '                "REVIEWED_PLAN_PATHS")\n'
        "        files += [args.plan, args.review]\n"
        "    from seed202_pilot.qualified_bridge import evidence_paths\n",
    )
    path.write_text(text, encoding="utf-8", newline="\n")
