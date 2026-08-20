# Open-loop execution + markerless continuous perception for the friction fit

## Repo state as of 2026-08-14

Checked against both remotes, not against these documents.

**BabyRobotPredicator `main` (`b45ac97`)** — the perception stack is merged and
still moving: #66 `ZedRecorderSession`, #67 markerless pose estimation, #68
depth-free bundles, #71 stage-1 extrinsics check, #72 stage-3 fp16 + crop and the
visibility gate, #73 markerless scene capture, #75 the pre-5.3 recorder fix, #76
output-path corrections.

**predicators `master` (`52613d5`)** — #125 renamed process → residual dynamics,
#127 added a physics-command channel, #128 made the registry sweep opt-in and
added `phys_params` point scoring, #129 stopped motion-planning the Push contact
strokes, #130 required a topple to persist, #131 fixed Panda finger dynamics.

**Three gaps, all actionable:**

1. **The submodule pointer is stale.**
   `origin/master:submodules/BabyRobotPredicator` is `396094e` (#63), **eight PRs
   behind** BabyRobotPredicator `main`. **Predicators cannot see any markerless
   or recorder code.** Bumping it is the first concrete step for Step 2.
2. **This branch is 12 commits behind `origin/master`.**
3. **`exp_domino_real.yaml` will conflict on merge.** Master changed 59 lines of
   it (largely #131's finger retune); the working tree carries local edits to
   the same file. Resolve deliberately rather than taking either side wholesale.

The `ActionExecutor` protocol is intact on master — exactly `tasks_for` /
`after_reset` / `after_step`, at `pybullet_env.py:73`. Step 1's design applies as
written.

## Context

The Stage-6 experiment sets the agent's belief to `domino_planning_friction: 0.1`
against a real table nearer 0.5 and expects system-ID to recover the truth. It
cannot today, for reasons established by inspection of a completed run:

- **The fit scores simulator against simulator.** `compute_rollout_sse`
  (`code_sim_learning/rollout_objective.py:36`) is "Total per-step SSE between
  free-running rollouts and observations". Of ~229 recorded states, **6 were real
  camera looks**; the rest are the twin integrating PyBullet at
  `domino_true_friction = 0.5`. Minimising that recovers 0.5 by construction.
- **The scored scope is wrong for this question.** `_moving_feature_scope`
  (`agent_sdk/tools/synthesis.py:654`) admits anything whose observed span
  exceeds `settle_tol` — the commanded robot arm and the dominoes' colour
  channels included. That is deliberate for the global-fidelity report it serves;
  it is the wrong scope for identifying one physical parameter (§3.1).
- **The agent therefore declined to fit friction.** Its own sweep ranked friction
  0.083 (SSE 162) above the truth 0.5 (SSE 211), non-monotonically. **Declining
  was the correct read of that evidence.** We fix the evidence, not the decision.

Three things now reshape the fix:

- **Open-loop execution.** Today the twin simulates an option, ships it, then
  simulates the next — the arm idles through a BiRRT solve between every option.
- **Continuous recording.** The ZEDs record the whole execution rather than
  taking six looks.
- **Markerless pose estimation**, which is built and measured, not planned.

**Intended outcome:** one contiguous arm motion per episode, recorded end to end,
post-processed into a dense real pose track, and scored over a scope that can
carry physics signal.

### Markers are no longer an option, so this is not a preference

The 20 mm ArUco markers are **not resolvable at HD720 at this camera distance** —
`cv2.aruco` finds 1 of ~7 on cam `30264679` and **0** on cam `32294776`
(`FINDINGS.md:226`). The marker pipeline cannot be run on these frames at all.
Markerless is the only path that produces poses from this footage.

---

## Step 1 — Open-loop episode execution

The enabler for Step 2, and a modest latency win in its own right.

**Size the latency win before spending effort here.** #129 made the Push contact
strokes step IK directly instead of calling BiRRT, taking **planner calls per
Push from 4 to 2** — so the per-option idle is roughly halved on the skill that
dominates a cascade episode. Step 1's case rests mainly on Step 2 needing one
contiguous recorded motion. Re-time an episode after the merge.

**Why it is safe.** With `real_robot_observe_at_option_boundary: False`,
`execute_chunks(..., observe=False)` returns `[]`, the absorb loop never runs, and
`after_step` returns `obs` **unchanged**. Shipping is a pure write-only side
effect, so *when* it happens is unobservable to the rollout — deferring every
chunk to the end produces a bit-identical twin trajectory. Everything reading
state mid-episode (`subgoal_annotations` monitor,
`agent_bilevel_max_execution_replans`, `terminate_on_goal_reached`) reads the
twin's own deterministic simulation either way.

**No protocol change is needed.** `execute_chunks` already packs a list of chunks
into one `StepRequest` (`real_robot_bridge.py:176-185`), and `_split_actions` is
"Stateless ACROSS calls… `RealRobot` drops that redundant command session-wide,
which is what makes per-chunk shipping safe" — the gripper dedup already handles
a batch.

**The post-processing latency makes this mandatory, not merely nice.** The
markerless pipeline runs at **~3.2× real time** (3.7 min on a 70 s take,
`FINDINGS.md:519`), so minutes of post-processing cannot come back inside an
episode under any design. Execution must not depend on one.

### Changes

1. **`ActionExecutor.after_episode(completed: bool)`** — new fourth method on the
   port (`envs/pybullet_env.py:73`), default no-op. The port has only
   `tasks_for` / `after_reset` / `after_step` today, which is the sole reason
   shipping must happen inside `after_step`.
2. **`PyBulletEnv.finish_execution(completed)`** delegates when an executor is
   attached; `BaseEnv` gets a no-op so `cogman` stays env-agnostic.
3. **`OptionBoundaryBuffer` accumulates completed chunks**; under the flag
   `after_step` appends and returns `obs` immediately
   (`real_robot_executor.py:405-421`).
4. **`after_episode(completed=True)` ships the accumulated list** in one
   `execute_chunks` call, and stops the recording (§Step 2). Log the batch's
   start/end from **both** `time.monotonic_ns()` and `time.time_ns()`: the
   wall-clock stamp pairs with the recorder's own host stamp (§3.2), and the
   monotonic one is immune to an NTP step landing mid-episode.
5. **Call sites:** `cogman.py:307` (the `keep_failed_demos` early return) and
   `cogman.py:327` (normal exit). `env` is in scope in
   `run_episode_and_get_observations`.

### Flag

```python
# Ship the whole episode's motion in one batch at the end, instead of one
# option at a time as it is simulated. Mutually exclusive with
# real_robot_observe_at_option_boundary: a boundary look has to happen
# between the options it separates.
real_robot_open_loop_episode = False
```

Assert the mutual exclusion loudly at `attach_real_robot` rather than silently
dropping whichever the config asked for second.

### Discard on abnormal termination

**Decision to confirm before implementing.** If the rollout throws mid-episode the
buffer holds a partial plan — half a pick, or a transport with no place. Ship
`completed=True` only; on the exception paths pass `completed=False`, discard, and
log how many chunks were dropped (`OptionBoundaryBuffer.discard` already reports
losses). Shipping a partial plan blind is worse than not shipping.

### Safety

Batching removes every natural stopping point: the arm runs the entire plan with
the e-stop as the only intervention, where today a bad first option is visible
before the second ships. This is a real regression in supervisability and is the
main argument for keeping the flag off by default.

### Falls out for free

`note_external_state_change` is currently called at every boundary even when
`observe=False`, re-seeding `Wait`'s quiescence tally when nothing external
changed. Open-loop stops calling it, which is more correct.

---

## Step 2 — The markerless pipeline as the observation source

Built and merged: `svo_to_bundle.py` → `sam2_track.py` → `domino_fit.py`, driven
by `reconstruct_dominoes_markerless.py` / `run_markerless.sh`. Its architecture is
record-live / process-offline, which is exactly what Step 3 needs.

### Recording: `ZedRecorderSession`

`pose_estimation/record_zed_video.py:139` — "Open ZEDs once; start/stop many SVO
takes without re-init." Its lifecycle maps onto the Step 1 hooks almost exactly,
which is the second thing `after_episode` earns its keep for:

| session call | executor hook |
|---|---|
| `open()` — cameras + warmup grabs, idempotent | once at `attach_real_robot` |
| `start_take(stamp, max_frames)` → take dir | `after_reset` |
| `stop_take(...)` → writes `meta.json` | **`after_episode`** |
| `close()` | run teardown |

Opening once across the whole run matters: a learning cycle is many episodes, and
per-episode camera init and warmup would otherwise be paid every time.

Three settings to fix at integration:

- **`export_mp4=False, export_depth=False` in the loop.** `stop_take` can export
  depth inline, but that is minutes of work per take — running it inside the
  episode loop would serialise post-processing into execution and undo Step 1.
  Record only; `svo_to_bundle.py` does depth later, off the critical path.
- **`max_frames`** bounds per-episode disk. Less urgent now that #68 took bundles
  to 48 MB, but still the cheapest guard against a take that never stops.
- **Record at HD720/60, not 30.** `DEFAULT_FPS["HD720"] = 60` already, so this
  costs nothing to ask for — and §3.3 now needs every frame it can get. See the
  frame-quantisation problem there.
- **`meta.json` is the handoff**: `serials`, `timestamp_clock`, `sdk_version`,
  `host_elapsed_s`, and `errors`, and is what `svo_to_bundle.py` reads to attach
  `timestamp_ns` to each frame. Check `errors` after every take — a camera that
  dropped out mid-episode should invalidate that episode's track rather than
  silently yield a short one.

**Cross-camera pairing is assumed, not established.** The installed ZED SDK is
**3.8.2**, which has no timestamp-clock API, so `open()` does not call
`set_timestamp_clock` and `meta.json` reports `timestamp_clock: "SDK_DEFAULT"`.
Frame times come from `get_timestamp(IMAGE)` after grab.

What is solid: `_grab_loop` runs the cameras against a shared
`threading.Barrier`, and `stop_take` writes `sdk_version` and `host_elapsed_s`
against a `_take_t0_host` host-clock stamp taken at `start_take`. That host stamp
is directly comparable to `time.time_ns()` on the robot side, which is the bridge
§3.2 needs.

### The output is already the observation track

`reconstruct_dominoes_markerless.py:98-105,158,185,259` emits:

```json
{"frame": "robot_base", "units": "meters/radians", "n_frames": N,
 "frames": [{"index": i, "timestamp_ns": ns,
             "dominoes": [{"id": 0, "center_base_m": [x, y, z],
                           "yaw_base_rad": ..., "roll_base_rad": ...,
                           "quat_base_xyzw": [...], "fall_deg": ...}]}]}
```

Three properties matter, and all three are already true:

- **`timestamp_ns` per frame** — the alignment key Step 3 needs.
- **`yaw_base_rad` + `roll_base_rad`, in the robot base frame** — exactly the two
  orientation features the predicators domino type carries. **This retires the
  pitch problem by construction.** The marker path logged *"is pitched -5.3 deg,
  which the (yaw, roll) domino state cannot represent; dropping the pitch"*; the
  markerless fit is parameterised in the representable variables from the start.
- **`fall_deg` per domino per frame** — the topple angle Step 3 thresholds
  directly, with no derivation of our own.

So Step 3 consumes this file. We do not define a format, and there is no
`State.privileged` marker: in open-loop nothing corrects the twin, so no recorded
state is an observation.

### Measured accuracy — the gating question is answered

Against hand-measured ground truth on a physical grid, in the base frame, same
body convention (`FINDINGS.md:154`):

| | cam `30264679` | cam `32294776` |
|---|---|---|
| position (median) | 24.1 mm | 37.9 mm |
| long-axis orientation (median) | **1.03°** | 6.29° |
| yaw, fallen only (median) | 0.24° | 2.10° |
| coverage | 1298/1575 (82%) | **1583/1585 (99.9%)** |

Step 3 needs orientation error under roughly 5°. **On the better camera it is
1.03°** — comfortably under, so Step 3 is viable.

**Position error is calibration, and it cancels.** The offset reproduces across
takes to within 5.1 mm on different layouts, and measured *displacements* of two
picked-and-placed dominoes were 210.6 mm and 199.4 mm against 211.9 mm and
200.0 mm by hand — **1.3 mm and 0.6 mm**. So "did it move, and how far" is ~1 mm
while "where is it in the base frame" inherits ~25–38 mm of extrinsics error.
This is a further argument for the difference-valued residual in §3.3.

**Camera trade-off to settle.** `30264679` is 6× better on orientation but drops
18% of frames; `32294776` tracks 99.9% from a side-on view. Onset timing needs
both angular accuracy and unbroken coverage. Recommend running both and using
agreement as the validator rather than picking one blind.

### Operating it in a learning loop

**The one open blocker: initialization boxes need a human.** `init_boxes.py`
offers only `manual` (drag boxes in an OpenCV window) and `given`; its own
docstring says "**Both sources are human-driven, and neither is the diagram's VLM
step**", and restoring VLM is "the top-ranked improvement to this stage: it is
the only route to [unattended initialization]". A learning loop cannot pause for
a human each episode.

Two routes around it, cheapest first:

- **Replay one human pass.** `babyrobot scene capture` takes `--boxes-json` to
  reuse an earlier run's boxes — "which is what makes a re-capture unattended".
  If the layout is fixed across an experiment's episodes, one human pass covers
  the whole run and no code is needed. Try this first.
- **Project the twin's geometry.** `--source given --boxes` accepts "a JSON list
  of `[x0,y0,x1,y1]`, or a path to one… the scriptable path, and the one to use
  over SSH or in a batch run", so predicators need only *write that file*. The
  task specifies the initial layout and the twin holds every domino's pose, so
  the boxes come from projecting known 3D geometry through the extrinsics rather
  than from detecting anything — a projection function plus a JSON write.

Note the markerless capture path is **one camera** (`--camera <serial>`); the
second ZED's cloud is not fused.

**Settings that must be passed explicitly, because the defaults are wrong for a
cascade:**

- **`--min-visibility` in the 40–70% band** (§3.4).
- **`--imgsz` 256, 512 or 1024, never 640.** 640 is *faster* than 1024 and
  "silently catastrophic — mean IoU 0.623, worst 0.049, masks essentially
  vanishing, no error from the model", because it does not divide into Hiera's
  window partitioning. Do not let a config expose a free integer here.

**Disk and throughput are not constraints.** Bundles are 48 MB (depth is no
longer stored; the `.svo` is replayed and "the poses do not change"), and the
pipeline runs at ~3.2× real time — 3.7 min for a 2111-frame take, with stage 4,
not SAM-2, the bottleneck at 48%. A 60 s episode post-processes in ~3 min, so a
learning cycle can plausibly keep up with the robot even before using the
across-takes parallelism (~2 GB of 24 GB VRAM).

Two watch items: the stage-3 crop is auto-derived and its `at_crop_edge` report
"has not yet been seen firing on real data"; and **do not decimate** — at 15 fps
the last two onsets collapse to 1 frame apart and cascade order stops being
resolvable (§3.3).

---

## Step 3 — Score the fit against the track

### 3.1 Unpollute the scope

**What `_moving_feature_scope` does** (`synthesis.py:654`): it is a plain "did
this number change?" test. It walks every object and every feature of that
object's type, records the min and max value seen across all recorded states, and
puts the feature in scope when `max - min > code_sim_learning_rollout_settle_tol`.
There is no check on what the number *means* — no type filter, no kinematic
filter.

**This is deliberate, and it is not a bug.** The docstring says so plainly: "The
open-loop report scores global fidelity, so its scope is 'everything that moves'
— independent of the artifact's declared `RESIDUAL_FEATURES`, which describe rule
scope and may legitimately be empty." Wide scope is the point. An artifact that
declares no residual features must still get a fidelity number, and that is what
the fallback buys.

**The mismatch is one of purpose, not correctness.** A global-fidelity report
asks "does the twin reproduce the whole recorded trajectory?", for which scoring
the arm is right. We are asking a different question — "which friction value best
explains the dominoes?" — for which scoring the arm is wrong, because the arm is
commanded and reproduces near-identically at *every* friction value. So this
section is a deliberate narrowing for the sysID objective, not a correction to
the report. **The report's own default behaviour must not change.**

Two concrete costs of the wide scope, for our objective specifically:

1. **Dilution, roughly per-feature.** `compute_residual_scaling` normalises each
   linear feature by its observed span and each angular feature by π, so every
   in-scope feature contributes comparably regardless of units. The domino type
   is `["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"]`; the colour
   channels change when the env recolours a domino and `is_held` flips 0 → 1 on a
   pick, so four of nine domino features carry no physics at all — and the robot's
   features come on top of that. Friction's signal is a minority share of the SSE
   being minimised.
2. **Lost segmentation — the one that actually broke the run.** The scope is
   reused to segment the episode. `_run_rollout_residuals` says it outright:
   "Whole trajectories first (no scope -> no truncation) to derive the motion
   scope, then re-prep with it so the scored rollouts get the same settled-tail
   truncation and rest-point segmentation the system-ID fit uses." With the arm
   in scope the arm is essentially always moving, so **173 of 184 steps count as
   active and the longest quiet run is 6** against `segment_min_rest_steps = 10`.
   `split_at_rest_points` can never cut and the whole episode collapses into one
   segment — the mechanism behind the `[225, 1]` segmentation seen in the run.
   Dominoes-only gives 70/184 active, a longest quiet run of 40, and segments
   `[50, 49, 31]`. This consequence is a side effect the docstring does not claim
   as intended, and it is the strongest argument in this section.

**The change: add `code_sim_learning_rollout_scope_types: List[str] = []`**
(empty = today's behaviour, so the fidelity report is untouched); set
`["domino"]` for this experiment.

**Separately, worth raising with the report's author rather than silently
fixing:** the docstring says "everything that **moves**" and reuses "the same
'still moving' cutoff the settled-tail truncation uses", but the implementation
tests whether a value *changed*. For `r`, `g`, `b` and `is_held` those are not
the same concept — a colour channel does not move, and a settle tolerance is not
meaningful applied to a boolean. That looks like a genuine docstring/implementation
gap, but it affects an existing report's output, so it should be fixed on its own
merits and not folded into this experiment's flag.

### 3.2 Alignment — align on a detected event, not on a clock

A global wall-clock↔sim-step map does **not** work: the arm's real duration is set
by the controller, while the sim advances a fixed
`pybullet_sim_steps_per_action / 240 = 83.3 ms` per action.

**But the cascade is passive physics, where both clocks run at the same physical
rate.** So align on an event visible in both streams — the start block's topple
onset — and take the offset from that. `timestamp_ns` gives the real side directly.

**Clock caveat.** On SDK 3.8.2 there is no process-wide timestamp clock;
`meta.json` reports `timestamp_clock: "SDK_DEFAULT"` and frame times come from
`get_timestamp(IMAGE)`. The bridge to robot-side time is `stop_take`'s
`host_elapsed_s`, measured against a `time.time()` stamp taken at `start_take`
and so directly comparable to `time.time_ns()` in the executor.

Because §3.2 takes its offset from a **detected event** rather than from the
clocks, none of this needs to be exact: the timestamps only have to bound the
search window well enough to find the right cascade. Logging both clocks at batch
start/end (Step 1, item 4) is sufficient for that.

### 3.3 The residual — propagation intervals

Score **per-domino topple onset relative to the first onset**: the inter-domino
propagation intervals. Three reasons:

- **Invariant to the alignment offset**, so residual error in §3.2 cancels.
- **Invariant to the calibration offset.** Position is off by 25–38 mm absolute
  but ~1 mm in differences; an onset is a threshold crossing on a *per-domino
  angle change*, so the constant frame error drops out. Scoring absolute
  positions would score the extrinsics.
- **Directly friction-sensitive**: friction sets propagation speed and whether
  the chain completes.

Reuse `rollout_objective.py:209-249` (`_onset_residuals`) with a threshold-crossing
detector beside the existing deviation detector.

**The sharpest constraint: the intervals are only a handful of frames long.** On
a real four-domino cascade the onsets are "separated by **6, 4 and 2 frames** at
30 fps" (`FINDINGS.md:581`) — 200, 133 and 67 ms, with the last one two frames. A
±1-frame detection error is then a **50% error on the quantity being fitted**,
which is the quantity friction is supposed to move.

Three responses, in order:

1. **Record at 60 fps.** `DEFAULT_FPS["HD720"]` is already 60 and the take
   measured above was 30, so this doubles every interval to 12, 8 and 4 frames
   for free, at ~6× real time. Do this before anything else in §3.3.
2. **Weight the intervals**, or fit the cumulative onset times rather than
   successive differences, so a single mis-detected onset does not dominate
   through a short denominator.
3. **Do not subsample.** 2× decimation was measured and rejected for exactly this
   reason: at 15 fps the last two onsets are 1 frame apart and "the cascade
   *order* stops being resolvable". Frame rate is not a throughput lever here.

Also worth checking before trusting an interval: whether a longer row spaces the
onsets further apart than this four-domino take did. If it does, the constraint
softens on its own.

### 3.4 Onset detection must survive spurious falls

Two false-onset mechanisms are measured, and the first fires exactly where §3.3
takes its reference point.

**Mechanism A — occlusion by the gripper.** On a real take, a **29.24° false
topple 15 frames before the real one**, on the very domino the arm was about to
push, at 34% mask visibility. Its fit residual was 7.1 mm — comfortably *inside*
`--max-resid`.
The point generalises: "a box fitted to a visible sliver is a confident fit to
real points. Residual measures how well the box explains the points it was given,
and says nothing about how much of the object those points represent."

This one is not incidental: the arm occluding the start block is a *guaranteed*
feature of every push episode, and the start block's onset is the reference point
§3.3 measures every other interval against.

**Handled upstream.** Stage 4 drops records below `--min-visibility` (default
0.6) of an object's median mask area, taking spurious readings from 3 to 0 while
keeping 100% of clean frames and leaving the cascade intact. Two settings must
not be moved: the denominator is **mask area, not `n_obs`** (`domino_fit`
subsamples to 4000 points, so `n_obs` saturates), and the working range is
**40–70%** — at 80% the gate deletes the real topple, which passes through 78–85%
visibility on its way down.

So the job here is to **pass `--min-visibility` explicitly** rather than inherit
a default, and to treat gate-dropped frames as missing data in the onset search
rather than as zeros.

**Mechanism B — orientation drift on an untouched domino.** On the place take a
domino that was never touched drifted from **4.4° ± 0.8 to 12.9° ± 1.6**, crossing
the 10° `Toppled` threshold; two others drifted 2° → 11–12°. Residuals stayed at
4–5 mm and no gate catches it (`FINDINGS.md:200`). Visibility does not explain it
— the affected object was not occluded.

**It may no longer reproduce.** #130 reports markerless perception of a real
episode where the placed block "never left the upright band before the push:
**0.20 deg median for the 30 s it stood placed**, peaking at 12.75 deg only while
the gripper occluded it, and zero frames past the 15 deg upright threshold
anywhere before the push." A 0.20° median over 30 s is not the drift signature,
and the one excursion is Mechanism A. Check directly whether the drift is
scene-specific before building defences against it.

**Mitigations:**

1. **Confirm, then backdate.** Only accept a fall once the angle passes an
   unambiguous value (~45–60°, far above any observed drift or occlusion
   artifact), then search backwards for the onset. The strongest mitigation, and
   there is a landed reference implementation to mirror rather than invent:
   `_TOPPLE_MIN_STEPS = 3` in `_topple_onset` (`cascade_certificate.py`) requires
   "3 consecutive non-held states at or past the threshold", with the run
   restarting on a carry or on returning inside the band. Note its asymmetry
   argument too — a run reaching the end of the episode counts at whatever
   length, because "trading this false reject for a false accept … is the worse
   direction for a certificate to fail in."
2. **Gate on visibility** — already implemented upstream (above). Note that an
   angular-rate or jump gate is *not* an alternative: jump gates were measured at
   20/40/60 mm and every variant "either broke the real cascade or discarded
   9–30% of good frames", because "during the real cascade the *other* dominoes
   translate 22–36 mm/frame, overlapping the 48–67 mm of the artifacts. **The
   separation is visibility, not speed.**" A real cascade *is* fast motion.
3. **Measure each domino against its own frame-0 angle**, never an absolute.

**One artifact survives all of the above**: "frame 1784, a 60 mm centre jump at
78% visibility… the case visibility alone cannot catch." The detector must
tolerate one bad frame without emitting an onset — which mitigation 1 does by
construction.

Note this threatens twin construction, not just the fit: a markerless
capture feeding `state_from_observation` could mark a standing domino `Toppled`,
which would silently disable `_canonical_start_yaw` (guarded on
`abs(roll) < fallen_threshold`) — the same guard interaction the 22° marker error
had, now with a measured cause.

**A third path to the same threshold.** ULTRA and NEURAL depth produce fall
angles differing "by a median of 0.5–4.2° and up to 15.8°, which is large next to
the 10° `Toppled` threshold: **the depth mode is not a free choice**"
(`FINDINGS.md:344`). The angle that decides `Toppled` therefore moves with a
setting, not only with the scene — another reason onsets must be detected from an
unambiguous fall (~45–60°) rather than from a threshold sitting inside the noise.
The caveat that NEURAL matters most on "painted or otherwise textureless
dominoes" applies directly if the dominoes get painted for tracking; re-run the
comparison on painted blocks before fixing a default.

### 3.5 The anchor stays at rest

Non-negotiable: a `State` carries pose but no velocity, so
*"resetting to a mid-cascade state discards the angular momentum that produced the
next step"* (`physical_sysid.py:6-9`; `states[0] at rest` at `rollout_env.py:19`,
enforced by `rollout_states` zeroing velocities). The free-run is anchored at the
last rest state before the push. Only *where residuals are taken* changes.

### 3.6 Flag

```python
# Score the free-running rollout against an external observation track
# (markerless per-frame poses) instead of against every recorded state.
code_sim_learning_rollout_score_observed_only = False
```

Flag on with no track available: log one WARNING and fall back to per-step scoring.
Scoring zero residuals would make every θ equally good and return the prior centre
with a confident-looking identifiability report — fail loud.

---

## Verification

1. **Open-loop, no hardware.** `real_robot_dry=True` with an injected duck-typed
   perception (`make_real_robot(perception=...)`, `real_robot_bridge.py:77-98`).
   Assert one `execute_chunks` call per episode, chunk count equals option count,
   order preserved, and a **bit-identical twin trajectory** versus the flag off.
   That equality is the whole safety argument for Step 1, so it should be explicit.

   **Take the baseline after merging master.** #131's finger dynamics and #129's
   Push phases both alter the twin's trajectories, so a baseline captured on this
   branch as it stands would not compare against anything.
2. **Discard path.** Force a mid-episode exception; assert nothing ships and the
   loss is logged, **and that `stop_take` still runs** — a recording left open
   would run until the disk fills. Recording teardown belongs in a `finally`,
   unlike chunk shipping which must not happen at all.
3. **Recorder lifecycle**, with a stubbed session: `open()` once across several
   episodes, one `start_take`/`stop_take` pair per episode, exports off, and a
   take whose `meta.json` carries `errors` marks its episode unusable.
4. **Onset detector against both spurious mechanisms.** Unit-test
   confirm-then-backdate on hand-built traces: a monotonic 4° → 13° drift with no
   fall, **a single-frame 29° occlusion spike 15 frames before a real topple**
   (the measured Mechanism A case), a real topple, a domino that never falls, one
   that falls before the push, and a trace with gate-dropped frames mid-fall. The
   drift and the spike must both yield no onset; the gapped trace must still yield
   the right one. `tests/envs/test_cascade_certificate.py` covers the sim-side
   analogue and is the model to follow.
5. **Synthetic recovery — the acceptance gate.** Pure sim `pybullet_domino`,
   `domino_true_friction=0.5`, base env at 0.1. Synthesise a track from the
   true-friction rollout at video frame rate, in the markerless JSON schema, fit
   with the flag on, assert it recovers ≈0.5; then fit the full recording and show
   it does not. Reproduces and refutes the 0.083 basin with no robot.

   Use `sim.residuals(rollout=True)` with `phys_params={name: value}`, which
   scores "ONE hypothesized physical-parameter point and report[s] the SSE ratio
   against the baseline — the composable primitive for agent-written targeted
   sweeps." The gate can then score 0.1 and 0.5 directly instead of running a
   full registry sweep per configuration. `sweep_params` and `phys_params` are
   mutually exclusive.

   That primitive's docstring reaches the same diagnosis as this plan from the
   other side — "an absolute rollout SSE is meaningless under chaotic replay
   divergence", and run_20260728_111805 "declined to declare on near-zero
   per-step residuals while the open-loop SSE ratio on the same data was ~340×"
   — but does not subsume it: a ratio computed over twin-generated states still
   ranks the twin's own friction best, which is the defect §3.3 addresses.
6. **Live.** Re-run with both flags on and read the identifiability sweep. Success
   is a monotone-ish sweep whose minimum sits near the real value — whether or not
   the agent then declares `PHYSICAL_PARAMS`.

## Also fix while here

- **`_persist_fit_trajectories`** (`agent_sim_learning_approach.py:1945`) only runs
  when the rollout branch is taken, so a run that declines to declare leaves no
  `fit_data/` to re-analyse. That is why this investigation worked from log text
  and why the 20260807 run cannot be re-fit offline. Persist whenever
  `code_sim_learning_persist_fit_data` is on.
- **Document that `code_sim_learning_num_mcmc_steps` does not affect the rollout
  sysID.** The rollout emcee branch was removed 2026-07
  (`physical_sysid.py:146-147`); the flag is read only by `fitting.py:178,575`.
  Our config sets it to 250 believing otherwise.

## Sequencing

**Do first, cheap and unblocking:**

1. **Merge `origin/master` into this branch** (12 behind), and **bump
   `submodules/BabyRobotPredicator` from `396094e` to `main`** (eight PRs of
   catch-up). Until the bump, predicators cannot import any markerless or
   recorder code, so no part of Step 2 can even be prototyped. Expect a conflict
   in `exp_domino_real.yaml` — master changed 59 lines of it, mostly #131's
   finger retune, against local edits to the same file.
2. **Re-time one episode.** #129 halved BiRRT calls per Push, so the idle gap
   Step 1 exists to remove may already be small. If it is, Step 1 becomes a
   Step-2 prerequisite rather than a win in its own right.

**Discard real cascade data recorded before #129.** That commit fixed a planner
detour that **struck the start block backwards** — measured at −102.7 mm and
−102.2 mm (away from the row) against +95 mm after the fix — and it reached
hardware: "the arm replays the twin's joint trajectory faithfully, so it
reproduced the backwards push exactly." Those episodes pushed the block the wrong
way and cannot serve as friction evidence. Re-record before fitting anything.

**Then, independent of perception:** Step 1 (open-loop) and Step 3.1 (scope) can
both land immediately — neither touches the camera path.

**Step 2's only prerequisite is unattended init boxes.** Try `--boxes-json`
replay first, since it is a flag rather than code; fall back to the twin
projection if the layout varies per episode.

**Before tuning against any number here**, do BabyRobotPredicator's own next-step
1: re-record with current extrinsics from a side-on viewpoint. That is a natural
first user of the `ZedRecorderSession` integration.

## Deliberately out of scope

Considered and rejected, so they do not get re-proposed:

- **Burst capture** (`observe_scene` wrapper, `frames=1`,
  `real_robot_cascade_samples`) — the markerless pipeline's record-then-process
  architecture supersedes it.
- **A `State.privileged` marker** and its downstream guards. In open-loop nothing
  corrects the twin, so no recorded state is an observation.
- **Defining an observation-track format, or storing raw-vs-snapped poses.** The
  pipeline already emits base-frame yaw/roll JSON with per-frame timestamps;
  `snap_stable` belongs to the marker path.
- **An angular-rate or jump gate on onset detection** — measured and rejected
  upstream; see §3.4.
- **Frame decimation as a throughput lever** — destroys cascade order; see §3.3.

## Open questions

- **Mid-fall accuracy is entirely unvalidated** and is where the friction signal
  lives.
- **Can init boxes be made unattended?** The only blocker left for a learning
  loop. `--boxes-json` replay is a flag away; twin projection is untried, and bad
  boxes are unrecoverable (emission 63% → 77% between careless and careful manual
  boxes).
- **Which depth mode**, given fall angles disagree by up to 15.8° and the choice
  is unresolved on painted blocks.
- **Does the orientation drift still reproduce?** The 4.4° → 12.9° measurement
  and the 0.20°-median episode disagree. Re-check directly before building
  defences. If it is real the cause is still open: exposure/lighting shifts
  partway through the take, and ~100 mm spacing with mutual occlusion.
- **Do the propagation intervals survive quantisation at 60 fps?** 12, 8 and 4
  frames is workable; if a longer row does not space the onsets further apart,
  the last interval stays marginal.
- **Do two cameras' SDK-default timestamps share an epoch on SDK 3.8.2?**
- **Batched motion is unsupervisable** between options; e-stop only. #129 raises
  the stakes: it fixed a bug where the arm confidently drove the block the wrong
  way, and open-loop is exactly the mode in which nobody sees that until the plan
  finishes.
