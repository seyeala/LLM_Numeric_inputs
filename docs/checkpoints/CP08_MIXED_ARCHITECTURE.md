# CP8 - Mixed text/numeric embedding architecture

Date: 2026-09-29
Repository: `F:\\LLM_Numeric_inputs`
Branch: `cp1-stabilization`
Starting commit: `4bad101`

## Scope

CP8 replaces the invalid legacy mixed-input pathway with an explicit embedding-sequence construction algorithm.
It does not run the minimum-number experiment; that remains CP9.

## Architecture

Mixed inputs use numeric markers in ordinary text:

```text
The measured value is $$12.5&&.
```

The wrapper parses each sample into text segments and numeric marker values.
Text segments are tokenized with the base tokenizer and mapped through the base model input embedding layer.
Each numeric marker is converted to a scalar tensor and projected through `input_projection`, producing one learned numeric embedding token.
The text embeddings and numeric embeddings are concatenated along the sequence dimension, not the hidden dimension.

This preserves the transformer hidden size:

```text
inputs_embeds: [batch, sequence, hidden_size]
attention_mask: [batch, sequence]
```

The transformer runs on `inputs_embeds`, and the output projection pools the final valid token using the existing attention-mask-aware pooling rule.

## Supported mixed input contract

```text
Input:  {"input_text": list[str] | str}
Output: Tensor[batch, 1]
```

CP8 supports:

- batch size 1,
- variable batch size,
- text with zero numeric markers,
- text with one numeric marker,
- text with multiple numeric markers,
- marker-only samples such as `$$9.5&&`,
- malformed-marker rejection.

## Implementation details

`NumericLMWrapper.forward_mixed` now builds a padded embedding batch. It no longer:

- joins multiple samples into one text string,
- concatenates text and numeric embeddings along hidden dimension,
- ignores parsed marker values,
- requires external `numeric_inputs` for mixed mode.

The mixed code uses the model input embedding layer from, in order:

1. `model.get_input_embeddings()` when available,
2. `model.embedding`,
3. `model.embed_tokens`,
4. `model.transformer.wte`.

## Compatibility script

`alignmentMixedN.py` is active again and uses `InputMode.MIXED_TO_NUMERIC`.
It remains a lightweight compatibility training entry point; CP9 will add the actual minimum-task mixed experiment.

## Tests added

`tests/test_wrapper_mixed.py` validates:

- one-marker batch output shape,
- batch-size-one behavior,
- multiple markers and hidden-size preservation,
- marker-only inputs,
- zero-marker text inputs,
- malformed marker failures,
- empty batch failure,
- one optimizer step updates projection weights but leaves transformer weights frozen,
- tokenizer requirement for text segments.

`tests/test_entrypoints.py` was updated so the mixed entrypoint is no longer expected to report CP3 deferral.

## Validation commands run

```powershell
.\.venv\Scripts\python.exe -m py_compile wrapperNM.py alignmentNN.py alignmentTextN.py alignmentMixedN.py scripts\environment_report.py scripts\cp3_smoke_contracts.py scripts\generate_dataset.py scripts\validate_dataset.py scripts\train_numeric_alignment.py scripts\evaluate_numeric_alignment.py scripts\train_text_alignment.py scripts\evaluate_text_alignment.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\ruff.exe check src scripts tests alignmentNN.py alignmentTextN.py alignmentMixedN.py wrapperNM.py dataset\text_numeric_generator_for_llm.py
```

## Validation result

```text
py_compile: ok
96 passed in 14.15s
No broken requirements found.
All checks passed!
```

## Acceptance checklist

- [x] Mixed mode preserves model hidden dimension.
- [x] Batch size is preserved end to end.
- [x] Multiple numeric markers in one sample produce multiple numeric sequence positions.
- [x] Mixed forward returns `[batch, 1]`.
- [x] One-step mixed training test produces finite gradients and updates projection weights.
- [x] Transformer weights remain frozen in the default mixed training test.
- [x] Malformed numeric markers are rejected.
- [x] Text-only mixed inputs are supported.
- [x] Marker-only mixed inputs are supported.
- [x] Root mixed alignment script uses the repaired mixed architecture.

## Known limitations after CP8

- CP8 validates the architecture, not a task-level mixed experiment.
- CP9 still needs to train/evaluate the mixed architecture on the minimum-number task with baselines.
- Mixed text generation has no template-holdout protocol yet; that belongs to CP9.

## Next checkpoint

CP9 should use the repaired mixed architecture for the minimum-number task and compare against a text-only baseline.
