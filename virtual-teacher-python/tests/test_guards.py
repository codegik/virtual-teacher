"""Tests about the setup: things that fail silently when two places stop agreeing."""
import os
import re
import sqlite3
import subprocess
from pathlib import Path

from pipo import config
from pipo.store import MIGRATIONS, Store
from pipo.values import load_kids, load_values

ROOT = Path(__file__).resolve().parents[1]


def test_the_model_file_the_app_loads_is_the_one_setup_downloads():
    setup = (ROOT / "scripts/setup.sh").read_text()
    m = re.search(r'^QWEN_FILE="([^"]+)"', setup, re.M)
    assert m and m.group(1) == config.QWEN_FILE


def test_the_venv_cuda_libraries_the_makefile_exposes_exist():
    # Without libcudart on the path, llama.cpp fails to import at all.
    makefile = (ROOT / "Makefile").read_text()
    for lib in re.findall(r"\$\(NV\)/([a-z_]+)/lib", makefile):
        assert (ROOT / ".venv/lib/python3.12/site-packages/nvidia" / lib / "lib").is_dir(), lib
    assert set(re.findall(r"\$\(NV\)/([a-z_]+)/lib", makefile)) == {"cuda_runtime", "cublas", "cudnn"}


def test_the_parents_files_are_shared_with_the_node_app():
    assert config.VALUES_DIR == ROOT.parent / "values"
    assert [k.id for k in load_kids()] == ["crianca-4", "crianca-9"]
    assert "<!--" not in load_values()


def test_opening_the_database_twice_runs_each_migration_once(tmp_path):
    db = tmp_path / "pipo.db"
    Store(db).start_session("crianca-9", __import__("datetime").datetime(2026, 10, 3))
    again = Store(db)
    assert again.schema_version() == len(MIGRATIONS) == 1
    count = sqlite3.connect(db).execute("select count(*) from sessions").fetchone()[0]
    assert count == 1


def test_a_transcript_line_is_stored_once_per_item(tmp_path):
    from datetime import datetime

    s = Store(tmp_path / "pipo.db")
    sid = s.start_session("crianca-9", datetime(2026, 10, 3))
    assert s.add_line(sid, "x_in", "crianca", "Oi", datetime(2026, 10, 3)) is True
    assert s.add_line(sid, "x_in", "crianca", "Oi de novo", datetime(2026, 10, 3)) is False
    assert s.lines(sid) == [("crianca", "Oi")]


def test_files_run_in_parallel_on_several_workers():
    assert int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1")) >= 2


def test_no_test_wipes_the_database():
    wipes = re.compile(r"delete\s+from\s+\w+\s*(;|\"|')|drop\s+table|truncate", re.I)
    for f in (ROOT / "tests").glob("*.py"):
        if f.name != Path(__file__).name:
            assert not wipes.search(f.read_text()), f.name


def test_setup_check_reports_missing_models_without_downloading(tmp_path):
    # Run the real script against a sandbox copy with a stub curl that logs its calls.
    sandbox = tmp_path / "proj"
    (sandbox / "scripts").mkdir(parents=True)
    (sandbox / "scripts/setup.sh").write_text((ROOT / "scripts/setup.sh").read_text())
    (sandbox / "scripts/setup.sh").chmod(0o755)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    for name in ("curl", "uv", "pactl"):
        (bin_ / name).write_text(f'#!/bin/sh\necho "{name} $*" >> {tmp_path}/calls\n')
        (bin_ / name).chmod(0o755)
    r = subprocess.run([str(sandbox / "scripts/setup.sh"), "--check"], capture_output=True, text=True,
                       env={"PATH": f"{bin_}:/usr/bin:/bin"})
    assert r.returncode == 1
    assert "todo qwen2.5-7b-instruct-q4_k_m.gguf not downloaded" in r.stdout
    calls = (tmp_path / "calls").read_text() if (tmp_path / "calls").exists() else ""
    assert "curl" not in calls and "uv sync" not in calls
