# Physical system identification pipeline

`../slides/sysid_pipeline_slides.html` is a self-contained reveal.js deck explaining the sysID stack end to end: why it exists, every pipeline stage with the implementing module, the measured failure modes of run_20260724_232411 and its neighbours, the honesty fixes that landed on `master` in PRs #99-#103, the 2026-07-29 result batch, honest limits, and the stage-2 roadmap.
Open it in a browser; press S for speaker notes.
It references both figures below as `../sysid/*.png`, so keep the deck and this directory in the same tree.

`sysid_pipeline.png` is the one-page diagram of the same pipeline (data collection, fit orchestration in `predicators/code_sim_learning/`, uncertainty accounting, consumers), with green badges on the six honesty fixes and the run that motivated each.

`sysid_landscapes.png` shows measured replay-SSE landscapes for four recordings at the true friction: which interaction data identifies the parameter (slide-rich pushes) and which cannot (pure topples with deterministic chaos spikes, carry motion).

Regenerate the figures with:

```bash
PYTHONPATH=. python docs/sysid/make_sysid_pipeline_fig.py
PYTHONPATH=. python docs/sysid/make_sysid_landscape_fig.py
```
