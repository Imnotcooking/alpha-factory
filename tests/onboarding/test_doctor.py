from __future__ import annotations

from pathlib import Path

from oqp.demo.doctor import doctor_exit_code, run_doctor
from oqp.demo.seed import seed_demo_profile


def test_doctor_recognizes_initialized_demo_profile(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src" / "oqp").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    seed_demo_profile(root, as_of="2026-07-17")

    checks = run_doctor(root)
    by_name = {check.name: check for check in checks}

    assert by_name["Runtime profile"].status == "pass"
    assert by_name["Demo fixtures"].status == "pass"
    assert doctor_exit_code(checks) == 0
