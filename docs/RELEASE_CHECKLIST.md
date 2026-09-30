# Release checklist

Use this checklist before tagging or merging a release branch.

- [ ] `python -m pip check` passes.
- [ ] `ruff check src scripts tests alignmentNN.py alignmentTextN.py alignmentMixedN.py wrapperNM.py dataset/text_numeric_generator_for_llm.py` passes.
- [ ] `pytest -q` passes offline.
- [ ] CP closure notes exist for CP1-CP11.
- [ ] README links to current setup, data, experiment, and benchmark workflows.
- [ ] Historical observations are separated from reproduced benchmark results.
- [ ] Generated artifacts are ignored or intentionally tracked as fixtures.
- [ ] Run artifacts needed for claims are reproducible from documented commands.
- [ ] Any remaining limitations are documented in the relevant checkpoint note.

Do not tag a release from a dirty working tree.
