import copy

from suite_tools.model_config import (
    config_inventory,
    describe_config,
    expand_model_keys,
    load_suite_config,
    main as model_config_main,
    render_module_config,
    resolve_agents,
    validate_suite_config,
)
from suite_tools.provider_client import ANTHROPIC_PRICE_PER_TOKEN


def test_render_sus_uses_panel_and_generic_local_endpoint():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "sus",
        judge_set="frontier",
        model_selector="group:local_endpoint_smoke",
    )

    assert rendered["judge_panel"] == [
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.7",
        "google/gemini-3.1-pro-preview",
    ]
    assert rendered["models"][0]["id"] == "local/example-model"
    assert rendered["models"][0]["label"] == "Local OpenAI-Compatible Endpoint"
    assert rendered["models"][0]["base_url"].endswith("/v1/chat/completions")
    assert rendered["models"][0]["max_parallel"] >= 1


def test_rendered_models_always_have_stable_condition_identity():
    config = load_suite_config()

    rendered_sus = render_module_config(
        config, "sus", judge_set="calibration", model_selector="gemini-flash"
    )
    rendered_aita = render_module_config(
        config, "aita", judge_set="calibration", model_selector="gemini-flash"
    )

    sus_model = rendered_sus["models"][0]
    aita_model = rendered_aita["models"]["gemini-flash"]
    assert sus_model["condition_id"] == "gemini-flash"
    assert aita_model["condition_id"] == "gemini-flash"
    assert sus_model["condition_hash"].startswith("sha256:")
    assert sus_model["condition_hash"] == aita_model["condition_hash"]


def test_resolved_route_changes_condition_identity_even_with_same_endpoint_name():
    config = load_suite_config()
    moved = copy.deepcopy(config)
    moved["endpoints"]["openrouter"]["openai_base_url"] = "https://proxy.example/v1"
    moved["endpoints"]["openrouter"]["chat_completions_url"] = (
        "https://proxy.example/v1/chat/completions"
    )

    original = render_module_config(
        config, "aita", judge_set="calibration", model_selector="gemini-flash"
    )["models"]["gemini-flash"]
    changed = render_module_config(
        moved, "aita", judge_set="calibration", model_selector="gemini-flash"
    )["models"]["gemini-flash"]

    assert original["route_hash"].startswith("sha256:")
    assert original["route_hash"] != changed["route_hash"]
    assert original["condition_hash"] != changed["condition_hash"]


def test_route_query_changes_condition_identity():
    config = load_suite_config()
    first = copy.deepcopy(config)
    second = copy.deepcopy(config)
    first["endpoints"]["openrouter"]["openai_base_url"] += "?api-version=a"
    second["endpoints"]["openrouter"]["openai_base_url"] += "?api-version=b"

    first_model = render_module_config(
        first, "aita", judge_set="calibration", model_selector="gemini-flash"
    )["models"]["gemini-flash"]
    second_model = render_module_config(
        second, "aita", judge_set="calibration", model_selector="gemini-flash"
    )["models"]["gemini-flash"]

    assert first_model["route_hash"] != second_model["route_hash"]
    assert first_model["condition_hash"] != second_model["condition_hash"]


def test_render_aita_uses_primary_judge_and_openai_base_url():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "aita",
        judge_set="calibration",
        model_selector="gpt-5-5",
    )

    assert rendered["judge"]["model_id"] == "google/gemini-3.1-pro-preview"
    assert rendered["judge"]["panel"] == ["google/gemini-3.1-pro-preview"]
    assert rendered["models"]["gpt-5-5"]["model_id"] == "openai/gpt-5.5"
    assert rendered["models"]["gpt-5-5"]["base_url"] == "https://openrouter.ai/api/v1"
    assert rendered["flip_generator"]["model_id"] == "google/gemini-3-flash-preview"


def test_agent_profiles_override_generator_models():
    config = load_suite_config()

    assert resolve_agents(config)["seeker"]["model_id"] == "google/gemini-3-flash-preview"
    assert resolve_agents(config, module="aita")["seeker"]["model_id"] == "google/gemini-3-flash-preview"
    assert resolve_agents(config, module="epis")["seeker"]["model_id"] == "google/gemini-3-flash-preview"

    rendered_sus = render_module_config(
        config,
        "sus",
        judge_set="calibration",
        model_selector="gemini-flash",
        agent_profile="gemini_35_flash",
    )
    rendered_aita = render_module_config(
        config,
        "aita",
        judge_set="calibration",
        model_selector="gemini-flash",
        agent_profile="sonnet_46",
    )

    assert rendered_sus["agent_profile"] == "gemini_35_flash"
    assert rendered_sus["analyzer"] == "google/gemini-3.5-flash"
    assert rendered_aita["agent_profile"] == "sonnet_46"
    assert rendered_aita["seeker"]["model_id"] == "anthropic/claude-sonnet-4.6"
    assert rendered_aita["flip_generator"]["model_id"] == "anthropic/claude-sonnet-4.6"


def test_render_direct_frontier_high_judge_configs():
    config = load_suite_config()

    rendered_sus = render_module_config(
        config,
        "sus",
        judge_set="direct_frontier_high",
        model_selector="gemini-flash",
    )
    rendered_aita = render_module_config(
        config,
        "aita",
        judge_set="direct_frontier_high",
        model_selector="gemini-flash",
    )

    assert rendered_sus["judge_panel"] == [
        "gpt-5.5",
        "claude-opus-4-7",
        "gemini-3.1-pro-preview",
    ]
    assert [judge["provider_api"] for judge in rendered_sus["judge_configs"]] == [
        "openai_compatible",
        "anthropic_messages",
        "gemini_generate_content",
    ]
    assert rendered_sus["judge_configs"][0]["api_key_env"] == "OPENAI_API_KEY"
    assert rendered_sus["judge_configs"][0]["request_options"]["max_tokens"] == 8192
    assert rendered_sus["judge_configs"][0]["request_options"]["reasoning_effort"] == "high"
    assert rendered_sus["judge_configs"][1]["api_key_env"] == "ANTHROPIC_API_KEY"
    assert rendered_sus["judge_configs"][1]["request_options"]["output_config"]["effort"] == "high"
    assert rendered_sus["judge_configs"][2]["api_key_env"] == "GEMINI_API_KEY"
    assert rendered_sus["judge_configs"][2]["request_options"]["generationConfig"]["maxOutputTokens"] == 8192
    assert (
        rendered_sus["judge_configs"][2]["request_options"]["generationConfig"]["thinkingConfig"]["thinkingLevel"]
        == "high"
    )
    assert rendered_aita["judge"]["model_id"] == "gpt-5.5"
    assert rendered_aita["judge"]["primary_config"]["api_key_env"] == "OPENAI_API_KEY"


def test_render_preserves_generic_endpoint_condition_metadata():
    config = load_suite_config()
    config["models"] = {
        **config["models"],
        "demo-served-endpoint": {
            "model_id": "provider/demo-model",
            "label": "Demo Served Endpoint",
            "endpoint": "openrouter",
            "served_profile_id": "safety-profile-v1",
            "served_profile_hash": "sha256:abc123",
            "condition_metadata": {"declared_by": "provider"},
            "request_options": {
                "reasoning": {"enabled": True, "exclude": False},
                "verbosity": "xhigh",
            },
        },
    }

    rendered_sus = render_module_config(
        config,
        "sus",
        judge_set="calibration",
        model_selector="demo-served-endpoint",
    )
    rendered_aita = render_module_config(
        config,
        "aita",
        judge_set="calibration",
        model_selector="demo-served-endpoint",
    )

    assert rendered_sus["models"][0]["served_profile_hash"] == "sha256:abc123"
    assert rendered_sus["models"][0]["condition_metadata"] == {"declared_by": "provider"}
    assert rendered_sus["models"][0]["request_options"]["verbosity"] == "xhigh"
    assert rendered_aita["models"]["demo-served-endpoint"]["served_profile_id"] == "safety-profile-v1"


def test_render_sus_supports_anthropic_native_endpoint():
    config = load_suite_config()
    config["models"] = {
        **config["models"],
        "claude-native-effort-high": {
            "model_id": "claude-opus-4-8",
            "label": "Claude Opus 4.8 native effort high",
            "endpoint": "anthropic_native",
            "condition_metadata": {
                "provider": "anthropic",
                "control_kind": "thinking_effort",
                "effort": "high",
            },
            "request_options": {
                "max_tokens": 4096,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"},
            },
        },
    }

    rendered = render_module_config(
        config,
        "sus",
        judge_set="calibration",
        model_selector="claude-native-effort-high",
    )

    model = rendered["models"][0]
    assert model["id"] == "claude-opus-4-8"
    assert model["base_url"] == "https://api.anthropic.com/v1/messages"
    assert model["provider_api"] == "anthropic_messages"
    assert model["api_key_env"] == "ANTHROPIC_API_KEY"
    assert model["request_options"]["output_config"]["effort"] == "high"


def test_native_anthropic_endpoint_renders_for_openai_compatible_modules():
    config = load_suite_config()
    config["models"] = {
        **config["models"],
        "claude-native-effort-high": {
            "model_id": "claude-opus-4-8",
            "label": "Claude Opus 4.8 native effort high",
            "endpoint": "anthropic_native",
            "condition_metadata": {
                "provider_route": "anthropic_native",
                "effort": "high",
            },
            "request_options": {
                "output_config": {"effort": "high"},
            },
        },
    }

    rendered = render_module_config(
        config,
        "aita",
        judge_set="calibration",
        model_selector="claude-native-effort-high",
    )

    model = rendered["models"]["claude-native-effort-high"]
    assert model["model_id"] == "claude-opus-4-8"
    assert model["base_url"] == "https://api.anthropic.com/v1/messages"
    assert model["provider_api"] == "anthropic_messages"
    assert model["condition_metadata"]["provider_route"] == "anthropic_native"
    assert model["request_options"]["output_config"]["effort"] == "high"


def test_all_native_anthropic_model_ids_have_dashboard_pricing():
    config = load_suite_config()
    model_ids = {
        str(model["model_id"]).split("/")[-1].replace(".", "-")
        for model in config["models"].values()
        if model.get("endpoint") == "anthropic_native"
    }

    assert model_ids - set(ANTHROPIC_PRICE_PER_TOKEN) == set()


def test_render_sus_supports_gemini_native_endpoint():
    config = load_suite_config()
    config["models"] = {
        **config["models"],
        "gemini-native-thinking-high": {
            "model_id": "gemini-3.1-pro-preview",
            "label": "Gemini native thinking high",
            "endpoint": "google_gemini_native",
            "condition_metadata": {
                "provider_route": "google_gemini_native",
                "effort": "high",
            },
            "request_options": {
                "generationConfig": {
                    "thinkingConfig": {"thinkingLevel": "HIGH"}
                }
            },
        },
    }

    rendered = render_module_config(
        config,
        "sus",
        judge_set="calibration",
        model_selector="gemini-native-thinking-high",
    )

    model = rendered["models"][0]
    assert model["id"] == "gemini-3.1-pro-preview"
    assert model["base_url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert model["provider_api"] == "gemini_generate_content"
    assert model["api_key_env"] == "GEMINI_API_KEY"
    assert model["request_options"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "HIGH"
    }


def test_native_gemini_endpoint_renders_for_openai_compatible_modules():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "aita",
        judge_set="calibration",
        model_selector="gemini-3-1-pro-native-high",
    )

    model = rendered["models"]["gemini-3-1-pro-native-high"]
    assert model["model_id"] == "gemini-3.1-pro-preview"
    assert model["base_url"] == "https://generativelanguage.googleapis.com/v1beta"
    assert model["provider_api"] == "gemini_generate_content"
    assert model["condition_metadata"]["provider_route"] == "google_gemini_native"
    assert model["condition_metadata"]["thinking_config_path"] == (
        "generationConfig.thinkingConfig.thinkingLevel"
    )
    assert model["request_options"]["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": False,
        "thinkingLevel": "high",
    }


def test_openrouter_gemini_high_renders_for_openai_compatible_modules():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "aita",
        judge_set="calibration",
        model_selector="gemini-3-1-pro-openrouter-high",
    )

    model = rendered["models"]["gemini-3-1-pro-openrouter-high"]
    assert model["model_id"] == "google/gemini-3.1-pro-preview"
    assert model["base_url"] == "https://openrouter.ai/api/v1"
    assert model["provider_api"] == "openai_compatible"
    assert model["api_key_env"] == "OPENROUTER_API_KEY"
    assert model["condition_metadata"]["provider_route"] == "openrouter"
    assert model["condition_metadata"]["effort"] == "high"
    assert model["request_options"]["reasoning"] == {
        "effort": "high",
        "exclude": True,
    }


def test_expand_model_group():
    config = load_suite_config()

    assert expand_model_keys(config, "group:calibration_smoke") == ["gemini-flash"]
    assert expand_model_keys(config, "group:frontier_03_04") == [
        "claude-opus-4-6",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "gpt-5-4",
        "gpt-5-5",
        "gemini-3-1-pro",
    ]
    assert expand_model_keys(config, "group:official_core_20260605") == [
        "gpt-chat-latest",
        "gpt-5-5-native-default",
        "gpt-5-5-native-high",
        "claude-opus-4-8-native-provider-default",
        "claude-opus-4-8-native-high",
        "claude-opus-4-8-native-xhigh",
        "gemini-3-1-pro",
        "gemini-3-1-pro-openrouter-high",
        "gemini-3-5-flash",
    ]
    assert expand_model_keys(config, "group:official_external_20260605") == [
        "deepseek-v4-pro",
        "deepseek-v3-2",
        "mistral-large-2512",
        "mistral-small-3-2",
        "grok-4-20",
        "qwen-3-6-plus",
        "glm-5-1",
        "kimi-k2-6",
        "gemma-4-31b",
        "nemotron-3-super",
        "mimo-v2-5-pro",
    ]
    assert expand_model_keys(config, "group:flash_35_smoke") == [
        "gemini-flash",
        "gemini-3-5-flash",
    ]
    assert expand_model_keys(config, "group:local_endpoint_smoke") == [
        "local-openai-compatible",
    ]
    assert expand_model_keys(config, "group:sus_xhigh_smoke") == [
        "gpt-5-5-xhigh",
        "claude-opus-4-8-xhigh",
    ]
    assert expand_model_keys(config, "group:provider_route_smoke") == [
        "gpt-5-5-high",
        "gpt-5-5-native-high",
        "claude-opus-4-8-high",
        "claude-opus-4-8-native-high",
    ]
    assert expand_model_keys(config, "group:claude_sonnet_5_native_effort_sus_n20") == [
        "claude-sonnet-5-native-low",
        "claude-sonnet-5-native-medium",
        "claude-sonnet-5-native-high",
        "claude-sonnet-5-native-xhigh",
        "claude-sonnet-5-native-max",
    ]
    assert expand_model_keys(config, "group:claude_fable_5_native_effort_sus_n20") == [
        "claude-fable-5-native-low",
        "claude-fable-5-native-medium",
        "claude-fable-5-native-high",
        "claude-fable-5-native-xhigh",
        "claude-fable-5-native-max",
    ]
    assert expand_model_keys(config, "group:gemini_native_route_smoke") == [
        "gemini-3-1-pro",
        "gemini-3-1-pro-openrouter-high",
        "gemini-3-1-pro-native-high",
        "gemini-3-5-flash",
        "gemini-3-5-flash-native-low",
    ]
    assert expand_model_keys(config, "group:gpt_5_6_sus_none") == [
        "gpt-5-6-sol-native-none",
        "gpt-5-6-terra-native-none",
        "gpt-5-6-luna-native-none",
    ]


def test_official_external_reasoning_starved_models_disable_reasoning():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "epis",
        judge_set="calibration",
        model_selector="group:official_external_20260605",
    )

    glm = rendered["models"]["glm-5-1"]
    kimi = rendered["models"]["kimi-k2-6"]
    for model in (glm, kimi):
        assert model["request_options"]["reasoning"] == {
            "effort": "none",
            "exclude": True,
        }
        assert model["condition_metadata"]["effort_policy"] == "explicit_openrouter_reasoning_none"
        assert model["condition_id"].endswith("-openrouter-no-reasoning")


def test_render_sus_xhigh_conditions():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "sus",
        judge_set="calibration",
        model_selector="group:sus_xhigh_smoke",
    )

    models = {model["label"]: model for model in rendered["models"]}
    gpt = models["GPT-5.5 / xhigh reasoning / OpenRouter"]
    opus = models["Claude Opus 4.8 / xhigh effort"]

    assert gpt["id"] == "openai/gpt-5.5"
    assert gpt["condition_metadata"]["effort"] == "xhigh"
    assert gpt["request_options"]["reasoning"]["effort"] == "xhigh"
    assert opus["id"] == "anthropic/claude-opus-4.8"
    assert opus["condition_metadata"]["effort"] == "xhigh"
    assert opus["request_options"]["verbosity"] == "xhigh"


def test_render_provider_route_smoke_preserves_provider_specific_controls():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "sus",
        judge_set="calibration",
        model_selector="group:provider_route_smoke",
    )
    models = {model["label"]: model for model in rendered["models"]}

    openrouter_gpt = models["GPT-5.5 / high reasoning / OpenRouter"]
    native_gpt = models["GPT-5.5 / high reasoning / OpenAI native"]
    openrouter_claude = models["Claude Opus 4.8 / default high effort"]
    native_claude = models["Claude Opus 4.8 / Anthropic native high effort"]

    assert openrouter_gpt["id"] == "openai/gpt-5.5"
    assert openrouter_gpt["condition_metadata"]["provider_route"] == "openrouter"
    assert openrouter_gpt["request_options"]["reasoning"]["effort"] == "high"

    assert native_gpt["id"] == "gpt-5.5"
    assert native_gpt["base_url"] == "https://api.openai.com/v1/chat/completions"
    assert native_gpt["condition_metadata"]["provider_route"] == "openai_native"
    assert native_gpt["request_options"]["max_tokens"] == 8192
    assert native_gpt["request_options"]["reasoning_effort"] == "high"

    assert openrouter_claude["request_options"]["verbosity"] == "high"
    assert native_claude["provider_api"] == "anthropic_messages"
    assert native_claude["request_options"]["output_config"]["effort"] == "high"


def test_render_sonnet_5_native_effort_group():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "sus",
        judge_set="direct_calibration_high",
        model_selector="group:claude_sonnet_5_native_effort_sus_n20",
    )

    efforts = [model["condition_metadata"]["effort"] for model in rendered["models"]]
    assert efforts == ["low", "medium", "high", "xhigh", "max"]
    assert {model["id"] for model in rendered["models"]} == {"claude-sonnet-5"}
    assert {model["provider_api"] for model in rendered["models"]} == {"anthropic_messages"}
    assert rendered["models"][-1]["max_parallel"] == 1
    assert rendered["models"][-1]["request_options"]["output_config"]["effort"] == "max"


def test_render_sonnet_5_native_uniform_128k_effort_group():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "sus",
        judge_set="frontier",
        model_selector="group:claude_sonnet_5_native_effort_uniform_128k_n20",
    )

    efforts = [model["condition_metadata"]["effort"] for model in rendered["models"]]
    assert efforts == ["low", "medium", "high", "xhigh", "max"]
    assert {model["id"] for model in rendered["models"]} == {"claude-sonnet-5"}
    assert {model["provider_api"] for model in rendered["models"]} == {
        "anthropic_messages"
    }
    for model in rendered["models"]:
        assert model["condition_id"].endswith("-128k")
        assert model["request_options"]["max_tokens"] == 128000
        assert model["request_options"]["thinking"] == {"type": "adaptive"}
        assert (
            model["request_options"]["output_config"]["effort"]
            == model["condition_metadata"]["effort"]
        )


def test_render_fable_5_native_effort_group():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "sus",
        judge_set="direct_calibration_high",
        model_selector="group:claude_fable_5_native_effort_sus_n20",
    )

    efforts = [model["condition_metadata"]["effort"] for model in rendered["models"]]
    assert efforts == ["low", "medium", "high", "xhigh", "max"]
    assert {model["id"] for model in rendered["models"]} == {"claude-fable-5"}
    assert {model["provider_api"] for model in rendered["models"]} == {"anthropic_messages"}
    for model in rendered["models"]:
        assert model["max_parallel"] == 1
        assert model["condition_metadata"]["provider_fallback"] == "disabled"
        assert model["request_options"]["max_tokens"] == 128000
        assert model["request_options"]["thinking"] == {
            "type": "adaptive",
            "display": "omitted",
        }
        assert model["request_options"]["output_config"]["effort"] == model["condition_metadata"]["effort"]


def test_render_gpt_5_6_sus_none_group():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "sus",
        judge_set="direct_calibration_high",
        model_selector="group:gpt_5_6_sus_none",
    )

    assert len(rendered["models"]) == 3
    tiers = [model["id"] for model in rendered["models"]]
    assert tiers == ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
    for model in rendered["models"]:
        assert model["condition_metadata"]["effort"] == "none"
        assert model["condition_metadata"]["effort_policy"] == "explicit_openai_reasoning_effort"
        assert model["request_options"]["reasoning_effort"] == "none"
        assert model["request_options"]["max_tokens"] == 128000
        # gpt-5.6 routes through the OpenAI Responses API (Chat Completions
        # rejects effort=max; Responses accepts the full none..max range).
        assert model["provider_api"] == "openai_responses"
        assert model["base_url"] == "https://api.openai.com/v1/responses"
        assert model["max_parallel"] == 2


def test_render_official_core_uses_native_frontier_thinking_conditions():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "sus",
        judge_set="calibration",
        model_selector="group:official_core_20260605",
    )
    models = {model["label"]: model for model in rendered["models"]}

    assert models["GPT-5.5 / high reasoning / OpenAI native"]["base_url"] == (
        "https://api.openai.com/v1/chat/completions"
    )
    assert models["Claude Opus 4.8 / Anthropic native high effort"]["provider_api"] == (
        "anthropic_messages"
    )
    assert models["Claude Opus 4.8 / Anthropic native xhigh effort"]["request_options"][
        "output_config"
    ]["effort"] == "xhigh"
    assert models["GPT Chat Latest / Instant"]["id"] == "openai/gpt-chat-latest"
    assert models["Gemini 3.1 Pro / OpenRouter high thinking"]["request_options"][
        "reasoning"
    ]["effort"] == "high"


def test_describe_config_lists_operable_selectors():
    config = load_suite_config()
    description = describe_config(config)
    private_product_token = "the" + "rry"

    assert "Judge sets:" in description
    assert "frontier: primary=openai/gpt-5.5" in description
    assert "calibration_smoke: gemini-flash" in description
    assert "flash_35_smoke: gemini-flash, gemini-3-5-flash" in description
    assert "local-openai-compatible: local/example-model (local_openai_compatible)" in description
    assert private_product_token not in description.lower()


def test_config_inventory_lists_operable_selectors_as_json_shape():
    inventory = config_inventory(load_suite_config())

    groups = {group["name"]: group for group in inventory["model_groups"]}
    models = {model["key"]: model for model in inventory["models"]}
    judges = {judge["name"]: judge for judge in inventory["judge_sets"]}

    assert inventory["schema_version"] == 1
    assert groups["calibration_smoke"]["models"] == ["gemini-flash"]
    assert models["gemini-flash"]["model_id"] == "google/gemini-3-flash-preview"
    assert models["gemini-flash"]["max_parallel"] == 3
    assert models["local-openai-compatible"]["endpoint"] == "local_openai_compatible"
    assert judges["calibration"]["panel"] == ["google/gemini-3.1-pro-preview"]
    profiles = {profile["name"]: profile for profile in inventory["agent_profiles"]}
    assert profiles["haiku_45"]["agents"][0]["model_id"] == "anthropic/claude-haiku-4.5"
    module_agents = {entry["module"]: entry for entry in inventory["module_agents"]}
    assert module_agents["aita"]["agents"][0]["model_id"] == "google/gemini-3-flash-preview"


def test_model_config_list_output_json_is_machine_readable(capsys):
    status = model_config_main(["--list", "--output-json"])
    out = capsys.readouterr().out

    assert status == 0
    assert '"model_groups"' in out
    assert '"calibration_smoke"' in out


def test_public_suite_config_has_no_private_product_entries():
    config = load_suite_config()
    private_product_token = "the" + "rry"

    serialized = "\n".join(
        [
            *config["endpoints"].keys(),
            *config["model_groups"].keys(),
            *config["models"].keys(),
            *[model.get("model_id", "") for model in config["models"].values()],
        ]
    ).lower()

    assert private_product_token not in serialized


def test_render_local_endpoint_smoke_group():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "aita",
        judge_set="calibration",
        model_selector="group:local_endpoint_smoke",
    )

    assert list(rendered["models"]) == ["local-openai-compatible"]
    model = rendered["models"]["local-openai-compatible"]
    assert model["model_id"] == "local/example-model"
    assert model["label"] == "Local OpenAI-Compatible Endpoint"
    assert model["base_url"] == "http://localhost:9999/v1"
    assert model["api_key_env"] == "LOCAL_OPENAI_COMPATIBLE_API_KEY"


def test_render_epis_frontier_exposes_full_judge_panel():
    config = load_suite_config()

    rendered = render_module_config(
        config,
        "epis",
        judge_set="frontier",
        model_selector="gemini-flash",
    )

    assert rendered["judge"]["model_id"] == "openai/gpt-5.5"
    assert rendered["judge"]["panel"] == [
        "openai/gpt-5.5",
        "anthropic/claude-opus-4.7",
        "google/gemini-3.1-pro-preview",
    ]


def test_validate_suite_config_accepts_current_config():
    assert validate_suite_config(load_suite_config()) == []


def test_validate_suite_config_rejects_unknown_group_model():
    config = load_suite_config()
    config["model_groups"] = {
        **config["model_groups"],
        "broken_group": ["does-not-exist"],
    }

    try:
        validate_suite_config(config)
    except ValueError as exc:
        assert "broken_group" in str(exc)
        assert "does-not-exist" in str(exc)
    else:
        raise AssertionError("expected invalid model group to fail validation")


def test_validate_suite_config_rejects_unknown_endpoint():
    config = load_suite_config()
    config["models"] = {
        **config["models"],
        "bad-endpoint": {
            "model_id": "example/bad-endpoint",
            "endpoint": "missing_endpoint",
        },
    }

    try:
        validate_suite_config(config)
    except ValueError as exc:
        assert "bad-endpoint" in str(exc)
        assert "missing_endpoint" in str(exc)
    else:
        raise AssertionError("expected invalid endpoint to fail validation")


def test_gpt_5_6_conditions_route_through_openai_responses_endpoint():
    config = load_suite_config()
    # aita renders models as a dict keyed by model key.
    rendered = render_module_config(
        config, module="aita", model_selector="group:gpt_5_6_sol_native_effort"
    )
    entry = rendered["models"]["gpt-5-6-sol-native-max"]
    assert entry["provider_api"] == "openai_responses"
    assert entry["base_url"] == "https://api.openai.com/v1/responses"
    assert entry["api_key_env"] == "OPENAI_API_KEY"
    assert entry["model_id"] == "gpt-5.6-sol"
    # effort metadata is preserved for the frozen grid.
    assert entry["condition_metadata"]["effort"] == "max"
    assert entry["request_options"]["reasoning_effort"] == "max"
    assert entry["request_options"]["max_tokens"] == 128000


def test_gpt_5_6_conditions_render_responses_base_url_for_sus_module():
    config = load_suite_config()
    rendered = render_module_config(
        config, module="sus", model_selector="group:gpt_5_6_sus_none"
    )
    # sus renders models keyed by model_id in the "id" field.
    entry = next(m for m in rendered["models"] if m["id"] == "gpt-5.6-sol")
    assert entry["provider_api"] == "openai_responses"
    assert entry["base_url"] == "https://api.openai.com/v1/responses"


def test_openai_responses_endpoint_validates():
    config = load_suite_config()
    # validate_suite_config must accept the openai_responses endpoint contract.
    assert isinstance(validate_suite_config(config), list)


def test_parent_profile_id_propagates_through_rendered_condition():
    """Regression: parent_profile_id declared in model YAML reaches the rendered condition."""
    config = load_suite_config()
    config["models"] = {
        **config["models"],
        "gpt-with-lineage": {
            "model_id": "openai/gpt-5.6-luna",
            "endpoint": "openrouter",
            "profile_id": "th-prompts-v3",
            "profile_hash": "f00f00",
            "parent_profile_id": "th-prompts-v2",
        },
    }

    rendered = render_module_config(
        config,
        "aita",
        judge_set="calibration",
        model_selector="gpt-with-lineage",
    )

    model = rendered["models"]["gpt-with-lineage"]
    assert model["parent_profile_id"] == "th-prompts-v2"
    assert model["profile_id"] == "th-prompts-v3"
    assert model["profile_hash"] == "f00f00"
