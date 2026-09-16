from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from deconversation import cli


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="the raw CLI currently imports the torch-backed solver eagerly",
)
def test_cli_raw_demo_writes_output(tmp_path):
    output = tmp_path / "result.csv"
    temp = tmp_path / "work"

    cli.main(
        [
            "--demo",
            "--mode",
            "raw",
            "--model",
            "unused",
            "--temp-output-dir",
            str(temp),
            "--output",
            str(output),
        ]
    )

    result = pd.read_csv(output, index_col=0)
    assert not result.empty


def test_cli_requires_reference_for_non_demo_run():
    with pytest.raises(SystemExit) as error:
        cli.main(["--bulk", "bulk.csv", "--model", "unused", "--mode", "raw"])

    assert error.value.code == 2


def test_cli_raw_demo_does_not_require_model(tmp_path):
    output = tmp_path / "result.csv"

    cli.main(["--demo", "--mode", "raw", "--output", str(output)])

    assert output.is_file()
