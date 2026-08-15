# Benchmark Model Nomenclature

Last reviewed: 2026-07-05

This note defines the compact model codes used in public benchmark result
views. It keeps score-map labels readable without hiding which model family or
served condition is being compared.

Scope boundary: this file is a display and decoding guide, not a run
configuration. The active public benchmark registry is `suite_models.yaml`.
Private overlays may add additional served endpoints, but those overlays should
not be copied into the public registry.

## Rule

The icon or mark carries the provider or owner. The text code carries the
public model family, version, and only the condition details needed to compare
results.

Examples:

- Gemini logo + `G-F-3` = Gemini Flash 3.
- Gemini logo + `G-P-3.1` = Gemini Pro 3.1.
- Claude/Anthropic logo + `C-O-4.6` = Claude Opus 4.6.
- Claude/Anthropic logo + `C-S-4.6` = Claude Sonnet 4.6.
- OpenAI logo + `GPT-5.5` = GPT-5.5.
- OpenAI logo + `CG-latest` = GPT Chat Latest / ChatGPT-style surface.
- OpenAI logo + `CX-5.5` = Codex-family surface, if present.
- Therapeutic Harness mark + `TH-GPT-5.5` = Therapeutic Harness configured
  with GPT-5.5 in the therapeutic response slot.

Provider initials are acceptable in the compact caption when they match the
user-facing model family: `C` for Claude, `CG` for GPT Chat surfaces, `CX` for
Codex, and `TH` for Therapeutic Harness. The full label and tooltip can carry
the rest.

## Code Grammar

| Segment | Meaning | Examples |
| --- | --- | --- |
| `G` | Gemini family | `G-F-3`, `G-P-3.1`, `G-FL-3.1` |
| `C` | Claude family | `C-O-4.8`, `C-S-5-native-high`, `C-F-5-native-max` |
| `GPT` | OpenAI GPT line | `GPT-5.5`, `GPT-5.5-xhigh` |
| `CG` | OpenAI chat-latest / ChatGPT line | `CG-latest`, `CG-4o-latest` |
| `CX` | OpenAI Codex line | `CX-5.5` |
| `Gm` | Google Gemma | `Gm-4-31B` |
| `K` | Kimi / Moonshot | `K-2.5`, `K-2.6` |
| `GLM` | Z.ai GLM | `GLM-5.1`, `GLM-5T` |
| `Q` | Qwen | `Q-3.6+` |
| `DS` | DeepSeek | `DS-V3.2`, `DS-V4P` |
| `X` | Grok / xAI | `X-4.20`, `X-4.1F` |
| `MI` | Xiaomi MiMo | `MI-2`, `MI-2.5` |
| `N` | Nemotron | `N-3S` |
| `M` | Mistral | `M-L-2512`, `M-S-3.2` |
| `L` | Local OpenAI-compatible endpoint | `L` |
| `TH` | Therapeutic Harness condition | `TH-Opus-4.7`, `TH-GPT-5.5`, `TH-Gemini-3.5F` |

## Therapeutic Harness Naming

For Therapeutic Harness records, use `TH-<base model code>` in tight square
labels and `therapeutic-harness/th-<base-model-slug>` for durable model ids.
The base model code is the therapeutic response slot, not private safety,
routing, or prompt composition.

Use this display convention:

| Surface | Format | Example |
| --- | --- | --- |
| Full display label | `Therapeutic Harness: <therapeutic model>` | `Therapeutic Harness: Claude Opus 4.6` |
| Compact square/chip | `TH-<model family/version>` | `TH-Opus-4.6` |
| Tooltip/detail | `Therapeutic Harness configured with <model> in the therapeutic response slot.` | `Therapeutic Harness configured with Claude Opus 4.6 in the therapeutic response slot.` |

Avoid `Pipeline + <model>` and parenthetical compact-code labels such as
`Therapeutic Harness (TH-Opus-4.7)` in new public surfaces. The colon label
reads as configuration: the model is inside the harness, not a peer system added
beside it.

Avoid one-letter durable TH slugs. Use full family words in model ids so they
do not collide with compact raw-model captions:

| Therapeutic slot | Durable TH slug | Compact code |
| --- | --- | --- |
| GPT-5.5, xhigh | `th-gpt-5-5-xhigh` | `TH-GPT-5.5-xhigh` |
| GPT-5.5 | `th-gpt-5-5` | `TH-GPT-5.5` |
| Claude Opus 4.7 | `th-opus-4-7` | `TH-Opus-4.7` |
| Gemini 3.5 Flash | `th-gemini-3-5-flash` | `TH-Gemini-3.5F` |
| Gemini 3.1 Pro | `th-gemini-3-1-pro` | `TH-Gemini-3.1P` |
| MiMo 2.5 Pro | `th-mimo-2-5-pro` | `TH-MiMo-2.5P` |

The public benchmark must treat Therapeutic Harness as an OpenAI-compatible
served endpoint. Benchmark artifacts should include the served model id,
condition hashes, response text, and public display label; they should not
include private prompt text, routing policy, safety-layer names, or internal
trace material.

## Active Public Registry

These are the model handles currently present in `suite_models.yaml`. Keep this
section synchronized with the registry. Do not add reference-only or private
served profiles here unless they are actually added to `suite_models.yaml`.

| Registry key | Model id / handle | Display label | Compact code |
| --- | --- | --- | --- |
| `claude-fable-5-native-high` | `claude-fable-5` | Claude Fable 5 / Anthropic native high effort | `C-F-5-native-high` |
| `claude-fable-5-native-low` | `claude-fable-5` | Claude Fable 5 / Anthropic native low effort | `C-F-5-native-low` |
| `claude-fable-5-native-max` | `claude-fable-5` | Claude Fable 5 / Anthropic native max effort | `C-F-5-native-max` |
| `claude-fable-5-native-medium` | `claude-fable-5` | Claude Fable 5 / Anthropic native medium effort | `C-F-5-native-medium` |
| `claude-fable-5-native-xhigh` | `claude-fable-5` | Claude Fable 5 / Anthropic native xhigh effort | `C-F-5-native-xhigh` |
| `claude-opus-4-6` | `anthropic/claude-opus-4.6` | Claude Opus 4.6 | `C-O-4.6` |
| `claude-opus-4-7` | `anthropic/claude-opus-4.7` | Claude Opus 4.7 / default high effort | `C-O-4.7-high` |
| `claude-opus-4-8-high` | `anthropic/claude-opus-4.8` | Claude Opus 4.8 / default high effort | `C-O-4.8-high` |
| `claude-opus-4-8-native-high` | `claude-opus-4-8` | Claude Opus 4.8 / Anthropic native high effort | `C-O-4.8-native-high` |
| `claude-opus-4-8-native-provider-default` | `claude-opus-4-8` | Claude Opus 4.8 / Anthropic native provider default | `C-O-4.8-native-default` |
| `claude-opus-4-8-native-xhigh` | `claude-opus-4-8` | Claude Opus 4.8 / Anthropic native xhigh effort | `C-O-4.8-native-xhigh` |
| `claude-opus-4-8-provider-default` | `anthropic/claude-opus-4.8` | Claude Opus 4.8 / OpenRouter provider default | `C-O-4.8-default` |
| `claude-opus-4-8-xhigh` | `anthropic/claude-opus-4.8` | Claude Opus 4.8 / xhigh effort | `C-O-4.8-xhigh` |
| `claude-sonnet-4-6` | `anthropic/claude-sonnet-4.6` | Claude Sonnet 4.6 | `C-S-4.6` |
| `claude-sonnet-5-native-high` | `claude-sonnet-5` | Claude Sonnet 5 / Anthropic native high effort | `C-S-5-native-high` |
| `claude-sonnet-5-native-low` | `claude-sonnet-5` | Claude Sonnet 5 / Anthropic native low effort | `C-S-5-native-low` |
| `claude-sonnet-5-native-max` | `claude-sonnet-5` | Claude Sonnet 5 / Anthropic native max effort | `C-S-5-native-max` |
| `claude-sonnet-5-native-medium` | `claude-sonnet-5` | Claude Sonnet 5 / Anthropic native medium effort | `C-S-5-native-medium` |
| `claude-sonnet-5-native-xhigh` | `claude-sonnet-5` | Claude Sonnet 5 / Anthropic native xhigh effort | `C-S-5-native-xhigh` |
| `deepseek-v3-2` | `deepseek/deepseek-v3.2` | DeepSeek V3.2 | `DS-V3.2` |
| `deepseek-v4-pro` | `deepseek/deepseek-v4-pro` | DeepSeek V4 Pro | `DS-V4P` |
| `gemini-3-1-pro` | `google/gemini-3.1-pro-preview` | Gemini 3.1 Pro | `G-P-3.1` |
| `gemini-3-1-pro-native-high` | `gemini-3.1-pro-preview` | Gemini 3.1 Pro / Google native high thinking | `G-P-3.1-native-high` |
| `gemini-3-1-pro-openrouter-high` | `google/gemini-3.1-pro-preview` | Gemini 3.1 Pro / OpenRouter high thinking | `G-P-3.1-openrouter-high` |
| `gemini-3-5-flash` | `google/gemini-3.5-flash` | Gemini 3.5 Flash | `G-F-3.5` |
| `gemini-3-5-flash-native-low` | `gemini-3.5-flash` | Gemini 3.5 Flash / Google native low thinking | `G-F-3.5-native-low` |
| `gemini-flash` | `google/gemini-3-flash-preview` | Gemini 3 Flash | `G-F-3` |
| `gemma-4-31b` | `google/gemma-4-31b-it` | Gemma 4 31B | `Gm-4-31B` |
| `glm-5-1` | `z-ai/glm-5.1` | GLM 5.1 / no reasoning | `GLM-5.1` |
| `gpt-5-4` | `openai/gpt-5.4` | GPT-5.4 | `GPT-5.4` |
| `gpt-5-5` | `openai/gpt-5.5` | GPT-5.5 / provider default / OpenRouter | `GPT-5.5-default` |
| `gpt-5-5-high` | `openai/gpt-5.5` | GPT-5.5 / high reasoning / OpenRouter | `GPT-5.5-high` |
| `gpt-5-5-native-default` | `gpt-5.5` | GPT-5.5 / provider default / OpenAI native | `GPT-5.5-native-default` |
| `gpt-5-5-native-high` | `gpt-5.5` | GPT-5.5 / high reasoning / OpenAI native | `GPT-5.5-native-high` |
| `gpt-5-5-xhigh` | `openai/gpt-5.5` | GPT-5.5 / xhigh reasoning / OpenRouter | `GPT-5.5-xhigh` |
| `gpt-chat-latest` | `openai/gpt-chat-latest` | GPT Chat Latest / Instant | `CG-latest` |
| `grok-4-20` | `x-ai/grok-4.20` | Grok 4.20 | `X-4.20` |
| `kimi-k2-6` | `moonshotai/kimi-k2.6` | Kimi K2.6 / no reasoning | `K-2.6` |
| `local-openai-compatible` | `local/example-model` | Local OpenAI-Compatible Endpoint | `L` |
| `mimo-v2-5-pro` | `xiaomi/mimo-v2.5-pro` | MiMo V2.5 Pro | `MI-2.5` |
| `mistral-large-2512` | `mistralai/mistral-large-2512` | Mistral Large 3 2512 | `M-L-2512` |
| `mistral-small-3-2` | `mistralai/mistral-small-3.2-24b-instruct` | Mistral Small 3.2 24B | `M-S-3.2` |
| `nemotron-3-super` | `nvidia/nemotron-3-super-120b-a12b` | Nemotron 3 Super 120B | `N-3S` |
| `qwen-3-6-plus` | `qwen/qwen3.6-plus` | Qwen 3.6 Plus | `Q-3.6+` |

## Gemini Native Thinking Notes

These notes apply to the Google-native Gemini entries in the active registry.
They are separate from OpenRouter Gemini entries because OpenRouter
`reasoning_effort` mappings can differ from the direct Gemini API.

Docs checked: 2026-06-08 against the Google Gemini API thinking guide,
generateContent API reference, and OpenAI compatibility guide. Live model
metadata also returned `200` for `models/gemini-3.1-pro-preview` and
`models/gemini-3.5-flash` on the `v1beta` Gemini API.

Direct Gemini route:

| Registry key | Native endpoint family | Native model id | Thinking field | Configured value | Documented default |
| --- | --- | --- | --- | --- | --- |
| `gemini-3-1-pro-native-high` | `generativelanguage.generateContent` | `gemini-3.1-pro-preview` | `generationConfig.thinkingConfig.thinkingLevel` | `high` | `high` |
| `gemini-3-5-flash-native-low` | `generativelanguage.generateContent` | `gemini-3.5-flash` | `generationConfig.thinkingConfig.thinkingLevel` | `low` | `medium` |

Current Google docs list these `thinkingLevel` options:

| Model family | Supported `thinkingLevel` values | Default if omitted | Disable thinking |
| --- | --- | --- | --- |
| Gemini 3.1 Pro | `low`, `medium`, `high` | `high` | Not supported |
| Gemini 3.5 Flash | `minimal`, `low`, `medium`, `high` | `medium` | Not fully supported; `minimal` is closest but not guaranteed off |
| Gemini 3 Flash | `minimal`, `low`, `medium`, `high` | `high` | Not fully supported; `minimal` is closest but not guaranteed off |
| Gemini 3.1 Flash-Lite | `minimal`, `low`, `medium`, `high` | `minimal` | Not fully supported |

Gemini 2.5-family models do not support `thinkingLevel`; use
`thinkingBudget` instead. The direct route should not use OpenRouter-style
`reasoning`, `reasoning_effort`, or `verbosity` fields. The native adapter
rejects those fields for `gemini_generate_content` configs.

For benchmark conditions, keep the configured thinking value explicit even
when it matches the provider default. This avoids ambiguity between provider
defaults, OpenRouter mappings, and direct Gemini API settings.

## Reference-Only Historical and Observed Models

Reference-only means these names help label old artifacts and decode drafts.
They are not active benchmark models unless the same id or explicit alias also
appears in `suite_models.yaml` and in the reviewed run plan for that suite.

| Model id / handle | Display label used in repo | Compact code | Notes |
| --- | --- | --- | --- |
| `anthropic/claude-3-opus` | Claude 3 Opus | `C-O-3` | Historical dashboard data. |
| `anthropic/claude-3.5-haiku` | Claude 3.5 Haiku | `C-H-3.5` | Historical dashboard data. |
| `anthropic/claude-3.7-sonnet` | Claude 3.7 Sonnet | `C-S-3.7` | Historical dashboard data. |
| `anthropic/claude-haiku-4.5` | Claude Haiku 4.5 | `C-H-4.5` | Shootout / dashboard artifacts. |
| `openai/chatgpt-4o-latest` | ChatGPT 4o Latest | `CG-4o-latest` | Historical SUS artifact. |
| `openai/gpt-4o-mini` | GPT-4o Mini | `GPT-4o-mini` | Historical dashboard data. |
| `openai/gpt-5.4-mini` | GPT-5.4 Mini | `GPT-5.4-mini` | Historical SUS artifact. |
| `google/gemini-3.1-flash-lite-preview` | Gemini 3.1 Flash Lite | `G-FL-3.1` | Dashboard data. |

## Implementation Note

The public results page derives compact square captions in
`suite_tools/public_results_page.py`. Keep that helper aligned with this file.
When adding a new model through the reviewed prepare-run flow, update
`suite_models.yaml` first and then verify:

1. Public display label.
2. Compact code.
3. Whether the square icon already provides provider context.
4. Whether this is a raw model, judge/seeker utility model, generic local
   endpoint, or Therapeutic Harness condition from a private overlay.

For publication pages, prefer showing only the records that actually feed the
current calculation. Use reference-only sections as legends for older or
diagnostic pages, not as the public benchmark roster.
