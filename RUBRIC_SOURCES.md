# Judge Rubric Source Registry

Last reviewed: 2026-05-11

These sources ground the judge-rubric calibration notes embedded in SUS, AITA,
and Epistemic judge prompts. They should not be treated as evidence about any
specific transcript. Judges must score only the observed transcript behavior
against the benchmark rubric.

## Source IDs

### NICE-CG178

- Source: NICE, "Psychosis and schizophrenia in adults: prevention and management"
- URL: https://www.nice.org.uk/guidance/cg178/chapter/Recommendations
- Rubric use: Supports the principle that psychosis-related support should
  preserve engagement while maintaining appropriate psychological and social
  intervention pathways.

### SAMHSA-FEP-CSC

- Source: SAMHSA, "First-Episode Psychosis and Co-Occurring Substance Use Disorders"
- URL: https://store.samhsa.gov/sites/default/files/d7/priv/pep19-pl-guide-3_0.pdf
- Rubric use: Grounds the distinction between validating distress and
  accepting false beliefs, and the emphasis on collaborative support that
  respects autonomy and expertise.

### MI-AUTONOMY-2012

- Source: Resnicow and McMaster, "Motivational Interviewing: moving from why to
  how with autonomy support"
- URL: https://link.springer.com/article/10.1186/1479-5868-9-19
- Rubric use: Grounds autonomy-supportive, person-centered guidance:
  reflective listening, eliciting change talk, exploring discrepancy, and
  avoiding coercive persuasion.

### AI-PSYCHOSIS-CONTEXT-2026

- Source: Nicholls et al., "\"AI Psychosis\" in Context: How Conversation
  History Shapes LLM Responses to Delusional Beliefs"
- URL: https://arxiv.org/abs/2604.13860
- Rubric use: Grounds the multi-turn concern that context can reveal whether a
  model inherits a user's worldview or continues to evaluate evidence. Useful
  for Epistemic delusion, stance amplification, and safety-context carryover.

### SUS-FRAMEWORK-V3

- Source: Internal framework document (not distributed); the operative scoring
  definitions are in `sus-bench/sus_bench/scoring_contract.py` and
  `docs/SCORING_MECHANICS.md`.
- Rubric use: Internal benchmark definition for Susceptibility to Unsafe
  Suggestion, unsafe-suggestion leakage, escalation resistance, safety-context
  retention, blinding, and multi-judge scoring.
