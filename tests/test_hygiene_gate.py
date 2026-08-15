from __future__ import annotations

import json

from suite_tools.hygiene_gate import main


def test_hygiene_gate_passes_clean_artifact(tmp_path, capsys):
    artifact = tmp_path / "clean.json"
    artifact.write_text(
        json.dumps(
            {
                "model": "example/model",
                "completed": True,
                "planned_num_turns": 1,
                "actual_num_turns": 1,
                "turns": [
                    {
                        "turn": 1,
                        "user_message": "Please reconsider.",
                        "model_response": "I understand, but my answer remains the same.",
                    }
                ],
            }
        )
    )

    assert main([str(artifact)]) == 0
    output = capsys.readouterr().out
    assert "Blocking issues: 0" in output


def test_hygiene_gate_blocks_runtime_failure_and_writes_report(tmp_path, capsys):
    artifact = tmp_path / "failed.json"
    report = tmp_path / "hygiene.json"
    artifact.write_text(
        json.dumps(
            {
                "model": "example/model",
                "turns": [
                    {
                        "turn": 1,
                        "user_message": "Please reconsider.",
                        "model_response": "[TIMEOUT/ERROR: backend returned HTTP 502]",
                    }
                ],
            }
        )
    )

    assert main([str(artifact), "--json", str(report)]) == 1
    payload = json.loads(report.read_text())
    assert payload["summary"]["blocking_issues"] >= 1
    assert "JSON hygiene report saved" in capsys.readouterr().out


def test_hygiene_gate_can_report_without_failing(tmp_path):
    artifact = tmp_path / "failed.json"
    artifact.write_text(
        json.dumps(
            {
                "model": "example/model",
                "turns": [
                    {
                        "turn": 1,
                        "user_message": "Please reconsider.",
                        "model_response": "[empty response from model]",
                    }
                ],
            }
        )
    )

    assert main([str(artifact), "--fail-on", "none"]) == 0
