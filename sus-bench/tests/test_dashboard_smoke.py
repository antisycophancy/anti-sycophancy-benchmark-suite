"""Smoke test for HTML dashboard rendering with judge panel data."""

from pathlib import Path

from sus_bench.report import write_html


def test_dashboard_renders_judge_panel(tmp_path):
    """Dashboard renders judge panel, bootstrap CI, and Cap/SUS labels."""
    agg = [{
        'model': 'test/model',
        'label': 'Test Model',
        'scenario': 'test',
        'scenario_name': 'Test Scenario',
        'runs': 3,
        'sus_mean': 25.0,
        'sus_ci_lower': 20.0,
        'sus_ci_upper': 30.0,
        'sus_bootstrap_ci_lower': 21.0,
        'sus_bootstrap_ci_upper': 29.0,
        'sus_bootstrap_mean': 25.0,
        'sus_stddev': 5.0,
        'irq_mean': 8.0,
        'pr_mean': 7.0,
        'er_mean': 9.0,
        'ca_mean': 6.0,
        'capitulation_phase': 'no capitulation',
        'individual_results': [{
            'run_number': 1,
            'capitulated': False,
            'score': {'sus': 25.0},
            'post_analysis': {
                'compliance_analysis': 'Model preserved the boundary.',
                'safety_theater_indicators': ['I understand your concern'],
                'judge_scores': [
                    {'judge': 'google/gemini-3-flash', 'irq': 8, 'pr': 7,
                     'er': 9, 'ca': 6, 'irq_explanation': 'Clean refusal'},
                    {'judge': 'anthropic/claude-sonnet-4.6', 'irq': 9, 'pr': 8,
                     'er': 9, 'ca': 7, 'irq_explanation': 'Strong refusal'},
                    {'judge': 'openai/gpt-5.4', 'irq': 7, 'pr': 6,
                     'er': 8, 'ca': 5, 'irq_explanation': 'Good refusal'},
                ],
                'num_judges': 3,
            },
            'conversation': [
                {'role': 'user', 'content': 'Test prompt'},
                {'role': 'assistant', 'content': 'Test response'},
            ],
        }],
    }]

    out = tmp_path / "dashboard.html"
    write_html(agg, out)
    content = out.read_text()

    assert 'Judge Panel' in content, 'Missing judge panel section'
    assert 'boot:' in content, 'Missing bootstrap CI'
    assert 'SUS Response=25.0' in content, 'Missing SUS Response summary'
    assert 'No Cap' in content, 'Missing Cap / No Cap run label'
    assert 'gemini-3-flash' in content, 'Missing judge name'
    assert 'Clean refusal' in content, 'Missing judge explanation'
    assert len(content) > 1000, 'HTML too short'


def test_dashboard_renders_without_optional_fields(tmp_path):
    """Dashboard renders without optional judge_scores."""
    agg = [{
        'model': 'test/model',
        'label': 'Test Model',
        'scenario': 'test',
        'scenario_name': 'Test Scenario',
        'runs': 1,
        'sus_mean': 50.0,
        'sus_ci_lower': 50.0,
        'sus_ci_upper': 50.0,
        'sus_stddev': 0,
        'irq_mean': 5.0,
        'pr_mean': 5.0,
        'er_mean': 5.0,
        'ca_mean': 5.0,
        'capitulation_phase': 'escalation',
        'individual_results': [{
            'run_number': 1,
            'capitulated': True,
            'score': {'sus': 50.0},
            'post_analysis': {
                'compliance_analysis': 'Capitulated under pressure.',
                'safety_theater_indicators': [],
            },
            'conversation': [
                {'role': 'user', 'content': 'Test prompt'},
                {'role': 'assistant', 'content': 'Test response'},
            ],
        }],
    }]

    out = tmp_path / "dashboard_minimal.html"
    write_html(agg, out)
    content = out.read_text()

    assert 'Test Model' in content
    assert 'Judge Panel' not in content  # no judge_scores
    assert 'Grade' not in content
    assert len(content) > 500
