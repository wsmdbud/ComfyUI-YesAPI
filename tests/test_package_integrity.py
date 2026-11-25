import json
import os
import subprocess
import pytest

import __init__ as root_pkg


def test_gitignore_blocks_config_json_and_no_secrets_in_source():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.run(
        ["git", "check-ignore", "config.json"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "config.json" in proc.stdout

    # Ensure config.example.json has only blank keys
    with open(os.path.join(repo_root, "config.example.json"), "r", encoding="utf-8") as f:
        cfg_ex = json.load(f)
    for k in ("default_api_key", "image_api_key", "llm_api_key", "video_api_key"):
        assert cfg_ex.get(k) == ""

    # Ensure no builtins tampering, pip install on import, or global SSL override in core/nodes/__init__.py
    for sub in ("__init__.py", "core", "nodes"):
        target = os.path.join(repo_root, sub)
        py_files = [target] if os.path.isfile(target) else [
            os.path.join(r, fn) for r, _, fns in os.walk(target) for fn in fns if fn.endswith(".py")
        ]
        for pyf in py_files:
            src = open(pyf, "r", encoding="utf-8").read()
            assert "_create_unverified_context" not in src
            assert "builtins." not in src
            assert "pip', 'install'" not in src and 'pip", "install"' not in src


def test_readme_documents_all_18_nodes_and_workflows_are_valid():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme_path = os.path.join(repo_root, "README.md")
    assert os.path.exists(readme_path)
    readme_text = open(readme_path, "r", encoding="utf-8").read()

    registered = set(root_pkg.NODE_CLASS_MAPPINGS.keys())
    assert len(registered) == 18
    for node_key in registered:
        assert node_key in readme_text, f"{node_key} missing from README.md"

    wf_dir = os.path.join(repo_root, "workflows")
    assert os.path.isdir(wf_dir)
    wf_files = sorted(f for f in os.listdir(wf_dir) if f.endswith(".json"))
    assert len(wf_files) >= 5

    referenced_yesapi_nodes = set()
    for wf_name in wf_files:
        wf_data = json.loads(open(os.path.join(wf_dir, wf_name), "r", encoding="utf-8").read())
        assert "nodes" in wf_data and isinstance(wf_data["nodes"], list)
        assert "links" in wf_data and isinstance(wf_data["links"], list)
        for n in wf_data["nodes"]:
            ntype = n.get("type", "")
            if ntype.startswith("YesAPI"):
                assert ntype in registered, f"Workflow {wf_name} uses unregistered node type {ntype}"
                referenced_yesapi_nodes.add(ntype)

    assert len(referenced_yesapi_nodes) >= 8


def test_no_node_mixes_aspect_ratio_and_resolution_in_single_dropdown():
    for node_key, cls in root_pkg.NODE_CLASS_MAPPINGS.items():
        it = cls.INPUT_TYPES()
        for sec in ("required", "optional"):
            for param_name, spec in it.get(sec, {}).items():
                opts = spec[0]
                if isinstance(opts, list):
                    has_ratio = any(isinstance(x, str) and ":" in x for x in opts)
                    has_res_or_px = any(
                        isinstance(x, str) and (x.upper().endswith(("K", "P")) or "x" in x.lower())
                        for x in opts
                    )
                    assert not (has_ratio and has_res_or_px), (
                        f"{node_key}.{param_name} mixes aspect ratios and resolutions: {opts}"
                    )
