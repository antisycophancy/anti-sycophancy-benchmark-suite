import json

from suite_tools.inspect_conversations import render_markdown


def test_render_sus_conversation_list(tmp_path):
    path = tmp_path / "sus-conversations.json"
    path.write_text(json.dumps([
        {
            "label": "Model A",
            "scenario": "bridge",
            "score": {"sus": 12},
            "conversation": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
    ]))

    text = render_markdown(path)

    assert "Model A | bridge" in text
    assert "**USER:**" in text
    assert "hello" in text
    assert '"sus": 12' in text


def test_render_turn_records_from_directory(tmp_path):
    path = tmp_path / "aita_item0_side_a.json"
    path.write_text(json.dumps({
        "model": "gpt-5-5",
        "side": "side_a",
        "item_idx": 0,
        "turns": [
            {"user_message": "AITA?", "model_response": "Maybe slow down."}
        ],
    }))

    text = render_markdown(tmp_path)

    assert "gpt-5-5 | side_a | item 0" in text
    assert "AITA?" in text
    assert "Maybe slow down." in text
