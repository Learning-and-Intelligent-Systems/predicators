"""Assemble a single-file reveal.js deck with base64-embedded figures."""
import base64
from pathlib import Path

HERE = Path(__file__).parent


def b64(name: str) -> str:
    return base64.b64encode((HERE / name).read_bytes()).decode()


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Reach-Limited Domino Tasks: Motivation &amp; Method</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/dist/reveal.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/dist/theme/white.css">
<style>
  .reveal { font-size: 30px; }
  .reveal h1 { font-size: 1.6em; }
  .reveal h2 { font-size: 1.15em; margin-bottom: 0.6em; }
  .reveal ul { font-size: 0.82em; line-height: 1.45; }
  .reveal li { margin: 0.28em 0; }
  .reveal table { font-size: 0.68em; margin: 0 auto; }
  .reveal table th, .reveal table td { padding: 0.3em 0.7em; }
  .reveal .small { font-size: 0.6em; color: #555; }
  .reveal .good { color: #1a7a1a; }
  .reveal .bad { color: #a01515; }
  .reveal .hl { color: #b3541e; font-weight: 600; }
  .reveal img { max-height: 540px; }
  .reveal blockquote { font-size: 0.75em; width: 92%; }
  .reveal code { color: #2a5d8a; }
</style>
</head>
<body>
<div class="reveal"><div class="slides">

<!-- 1 ─ Title -->
<section>
  <h1>Making Miscalibration Matter</h1>
  <p>Reach-limited <em>minimum-block</em> tasks for the domino domain</p>
  <p class="small">Motivation &amp; method of the new task generator<br>
  (base-sim system-identification experiment line)</p>
  <aside class="notes">Goal of this talk: explain why we replaced the domino
  task generator and how the new one works. Everything shown is implemented
  and verified in sim; baseline + oracle arms are ready to launch.</aside>
</section>

<!-- 2 ─ Why touch the tasks at all -->
<section>
  <h2>Why change the domino tasks at all?</h2>
  <ul>
    <li>Our agent learns <b>world models</b>: predicates + process rules + parameters</li>
    <li>Boil: hidden heat dynamics → a real learning problem ✓</li>
    <li>Domino: <span class="hl">all dynamics live in the base PyBullet sim</span> —
        the learned residual is a no-op → <b>nothing to learn</b></li>
    <li>Natural fix: learn the base sim's <b>physical parameters</b> (system identification)</li>
    <li class="fragment">…but first: <b>does a wrong parameter even hurt?</b></li>
  </ul>
  <aside class="notes">In earlier domino runs the agent invented no predicates
  and no rules — it just planned against a known simulator. The predicate set
  never grew past Holding. For domino to test world-model learning at all,
  the base sim itself must be learnable — and being wrong must have
  consequences.</aside>
</section>

<!-- 3 ─ Which parameter -->
<section>
  <h2>Which parameter could we learn? (measured)</h2>
  <table>
    <tr><th>parameter</th><th>swept</th><th>effect on cascade reach g<sub>crit</sub></th></tr>
    <tr><td><b>lateral friction</b></td><td>0.1 → 1.0</td>
        <td class="good"><b>0.11 → 0.16 m under the real push — dominant</b></td></tr>
    <tr><td>mass</td><td>0.02 → 0.5 (25×)</td><td class="bad">none (flat)</td></tr>
    <tr><td>restitution</td><td>0 → 0.6</td><td class="bad">none (flat)</td></tr>
  </table>
  <ul>
    <li>g<sub>crit</sub> = largest inter-domino gap the cascade still crosses</li>
    <li>Mass cancels in rigid toppling &amp; equal-mass collisions → a <em>null</em> target</li>
    <li><b>Friction is the system-identification target</b></li>
  </ul>
  <aside class="notes">Two probes: a gentle tilt probe (g_crit 0.06→0.11)
  and a faithful robot-Push probe (0.11→0.16, saturating above friction 0.5).
  Lesson: the faithful probe was necessary — the reduced probe overstated
  steepness. Mass being null is itself a useful negative result.</aside>
</section>

<!-- 4 ─ Money curve -->
<section>
  <h2>Miscalibration causes real execution failures</h2>
  <img src="data:image/png;base64,{{IMG_PHASE1}}" alt="g_crit vs friction; success collapse">
  <p class="small">A plan tuned to friction 0.5 (gap 0.16) topples 5/5 when real friction ≥ 0.5,
  but 1/5 when real friction &lt; 0.5. Failure is <b>directional</b>: over-estimating reach.</p>
  <aside class="notes">Left: reach rises with friction and saturates. Right:
  the same fixed plan executed at each "real" friction — collapses below the
  planning friction. This is the sim-to-"real" gap in miniature, with
  friction as the only difference.</aside>
</section>

<!-- 5 ─ Old tasks hide it -->
<section>
  <h2>…but the <em>old</em> tasks hide the gap</h2>
  <ul>
    <li>Old generator lays the whole solution chain at fixed <code>0.098&nbsp;m</code> spacing</li>
    <li>Reach bands under the real push:</li>
  </ul>
  <table>
    <tr><th>gap</th><th>outcome</th></tr>
    <tr><td>≤ 0.11 m</td><td class="good">topples at <b>every</b> friction (robust band) ← old tasks live here</td></tr>
    <tr><td>0.12 – 0.16 m</td><td class="hl">friction-sensitive band</td></tr>
    <tr><td>≥ 0.17 m</td><td class="bad">impossible at any friction</td></tr>
  </table>
  <ul>
    <li>⇒ any sim-valid plan also works in real, <b>at any friction</b> → sysID is moot</li>
    <li>Agents get generous blocks → never operate near the reach limit</li>
  </ul>
  <aside class="notes">Session-log evidence: solved plans used ~0.10 m gaps —
  "well within topple reach". Over-provisioning is why there was nothing to
  learn: the tasks never forced the model to be right.</aside>
</section>

<!-- 6 ─ LLM minimality problem -->
<section>
  <h2>Second gap: LLM planners don't minimize</h2>
  <ul>
    <li>Classical planners prefer short plans; <b>LLM planners have no such bias</b></li>
    <li>A cautious LLM <em>over-builds</em> → accidentally tight gaps → succeeds with a wrong model</li>
    <li>A soft “use few blocks” instruction has no teeth</li>
    <li>A hard block <em>budget</em> can't force failure either — the baseline can just use all of them</li>
    <li class="fragment"><b>⇒ the reward itself must encode the true minimum</b></li>
  </ul>
  <aside class="notes">This was a user-caught design point: our first
  formulation relied on plan-cost minimization that LLM agents simply don't
  have. Budgets alone fail both ways: cap at baseline's count and the correct
  denser solution can't fit; cap at the correct count and the baseline
  accidentally succeeds by using everything.</aside>
</section>

<!-- 7 ─ Pipeline overview -->
<section>
  <h2>The whole pipeline, in one slide</h2>
  <div style="display:flex; align-items:stretch; gap:0.35em; font-size:0.52em; text-align:left;">
    <div style="flex:1; background:#eef3f8; border:1.5px solid #2a5d8a; border-radius:8px; padding:0.6em;">
      <b>1&nbsp;·&nbsp;Sample geometry</b><br>
      green start → purple target.<br>
      <b>Straight</b>: span 0.13–0.30&nbsp;m.<br>
      <b>Turn</b> (~40%): entry + exit legs, one 90° corner.
    </div>
    <div style="align-self:center; font-size:1.6em;">→</div>
    <div style="flex:1; background:#eef8ee; border:1.5px solid #1a7a1a; border-radius:8px; padding:0.6em;">
      <b>2&nbsp;·&nbsp;Find true K*</b><br>
      simulate <em>real pushes</em> at true friction (µ=0.1).<br>
      Straight: evenly-spaced chain.<br>
      Turn: <em>search</em> corner layouts.
    </div>
    <div style="align-self:center; font-size:1.6em;">→</div>
    <div style="flex:1.15; background:#fdf1ec; border:1.5px solid #b3541e; border-radius:8px; padding:0.6em;">
      <b>3&nbsp;·&nbsp;Keep only tasks that separate the models</b><br>
      recompute believed&nbsp;K at planning friction (µ=0.5); keep only if the
      wrong model <b>under-counts</b>.<br>
      <span style="color:#a01515;">most attempts die here — that's the point</span>
    </div>
    <div style="align-self:center; font-size:1.6em;">→</div>
    <div style="flex:1; background:#f3eef8; border:1.5px solid #6a4a8a; border-radius:8px; padding:0.6em;">
      <b>4&nbsp;·&nbsp;Attach reward, stage blues</b><br>
      success ⇔ topple&nbsp;∧&nbsp;≤&nbsp;K* blues used.<br>
      4 blues staged (&gt;&nbsp;K*), so over-building is possible — and punished.
    </div>
    <div style="align-self:center; font-size:1.6em;">→</div>
    <div style="flex:0.8; background:#f5f5f5; border:1.5px solid #777; border-radius:8px; padding:0.6em;">
      <b>5&nbsp;·&nbsp;Cache</b><br>
      keyed by config + seed + <em>code digest</em>; auto-invalidates on any change.
    </div>
  </div>
  <ul style="margin-top:0.7em;">
    <li>Every kept task is a <b>constructive proof</b>: the K*-search's winning layout solves it,
        and a planner with the wrong friction provably under-builds it</li>
    <li class="small">Drop reasons: direct push already solves it (K*&lt;1) · needs every staged blue (no spare
        for the over-build check) · both frictions agree (dead band)</li>
  </ul>
  <aside class="notes">Read left to right: sample, measure, filter, arm,
  cache. Step 2 is honest physics — real Push rollouts with the push agents
  actually use, not geometry arithmetic. Step 3 is the differentiation
  filter, direction-aware for over/under-reach. Step 4 arms the
  DominoEvaluator (offline K*). Step 5: the cache key hashes every domino_/pybullet_/skill_phase_
  flag, the seed, task counts, and a digest of the domino env + skill source
  — cold gen ~45 s for 5 tasks, warm reload ~0.01 s.</aside>
</section>

<!-- 8 ─ The reward -->
<section>
  <h2>The minimum-block reward</h2>
  <blockquote><b>reward = 1[Toppled(target) ∧ legitimate cascade] − c · blocks_used</b><br>
  K* = true minimum #blues that topple the target — computed <b>by simulation at the true friction</b>, kept <b>offline-only</b> (metrics, never the criterion)</blockquote>
  <table>
    <tr><th>blues used</th><th>outcome (verified in sim)</th></tr>
    <tr><td>K* − 1</td><td class="bad">chain dies short — no topple, no bonus ✗</td></tr>
    <tr><td>K*</td><td class="good">topples ✓ reward 1 − c·K* (max)</td></tr>
    <tr><td>K* + 1</td><td class="bad">topples but wasteful — reward 1 − c·(K*+1)</td></tr>
  </table>
  <ul>
    <li>Under-build fails on physics; over-build pays the per-block cost — <b>two-sided</b>, no K* in the reward</li>
    <li>Maximizing reward ⇔ having a <b>calibrated reach model</b> — that's the whole point</li>
    <li><code>blocks_used</code> = toppled movable blues in the final state (plan-free, state-based)</li>
  </ul>
  <aside class="notes">Implemented as a per-task evaluator
  (EnvironmentTask.evaluator, a DominoEvaluator in the standard RL
  (reward, terminated) shape; terminated = the atom check, the legitimacy
  certificate gates the bonus). K* stays out of everything agent-visible,
  so the reward channel can't leak "fewer would have sufficed" — the agent
  must derive spacing from a physics model.</aside>
</section>


<!-- 9 ─ Anatomy figure -->
<section>
  <h2>One task, two models</h2>
  <img src="data:image/png;base64,{{IMG_ANATOMY}}" alt="task anatomy">
  <p class="small">The miscalibrated planner's 1-blue plan <em>validates in its own sim</em> —
  and dies in the real one. Unrecoverable: the fallen green start is not movable, so no replan can restart the cascade.</p>
  <aside class="notes">Numbers are the measured reaches. Note the failure is
  terminal — the restricted Push only targets the green start block, which
  has already fallen. Replanning is allowed (2 replans) and still can't
  recover, which makes the baseline stronger and the result cleaner.</aside>
</section>

<!-- 10 ─ Computing K* -->
<section>
  <h2>Computing K* honestly: simulate, don't count</h2>
  <ul>
    <li>All verification drives the <b>real robot Push</b> (IK failures ⇒ candidate is a miss)</li>
    <li><b>Straight tasks:</b> even spacing is optimal on a line → try k = 0, 1, 2, …</li>
    <li><b>Turn tasks (90°, dominoes only):</b> the evenly-spaced L is <span class="bad">not minimal</span> —
        sliding the turn pair toward the start (“stretched corner”) saves a block.
        <span class="hl">Our first K* was wrong; simulation caught it.</span></li>
    <li>⇒ K* = minimum over a <b>layout search family</b> of
        <span class="hl">agent-buildable</span> layouts:
      <ul>
        <li>straight-line probe (can the corner be cheated?)</li>
        <li>corner search: entry per-gap ∈ {0.10, 0.13, 0.15} + ONE natural-yaw
            corner blue from sim-calibrated (yaw, in-gap, out-gap) configs</li>
        <li class="bad">the generator's mirrored 45° pair is <b>excluded</b> —
            no planner would propose it, so K* must not assume it</li>
      </ul>
    </li>
    <li class="small">Geometric pruning (gaps outside (0.03, 0.20) skipped) bounds the sims/task</li>
  </ul>
  <aside class="notes">The stretched-corner discovery: a 2-blue re-placement
  toppled at BOTH frictions where our constructed 3-blue L was assigned as
  K*. That broke exact-minimality semantics and made baseline outcomes a
  coin flip. Searched K* is an upper bound over the family, but the family
  includes the strategies that dominate in practice. The natural-corner
  configs were calibrated from the oracle run's own solution (a -36 deg
  corner blue with uneven gaps) — the mirrored pair was dropped 2026-07-03
  on the agent-buildability principle.</aside>
</section>

<!-- 11 ─ Corner layout search, elaborated -->
<section>
  <h2>"Search corner layouts" — what K* actually tries</h2>
  <div style="display:flex; gap:0.8em; align-items:center;">
    <div style="flex:1.05; text-align:left;">
      <ul style="font-size:0.72em;">
        <li><b>Why search?</b> Around a corner, evenly spaced is <em>not</em>
            cheapest — sliding the corner along the entry leg
            (<span class="hl">"stretched corner"</span>) can save a whole
            block. Counting an even L over-states K*.</li>
        <li><b>Only agent-buildable layouts.</b> The family contains what a
            planner would actually propose; the generator's mirrored 45°
            pair is <span class="bad">excluded by principle</span>.</li>
        <li><b>For each k (ascending), try every candidate:</b>
          <ul>
            <li><b>straight-line probe</b> — k blues evenly spaced
                start→target, ignoring the corner
                (<em>can the corner be cheated?</em>)</li>
            <li><b>natural-corner family</b> — k1 entry blues (per-gap
                g ∈ {0.10, 0.13, 0.15} <em>slides the corner</em>), ONE
                corner blue facing 36–54° into the turn with sim-calibrated
                in/out gaps, exit blues evenly spaced</li>
          </ul>
        </li>
        <li>Every candidate = a <b>full PyBullet rollout</b> with a real
            push — no geometry arithmetic. The corner configs were
            calibrated from the <b>oracle run's own −36° corner blue</b></li>
        <li>First k with <em>any</em> toppling layout wins;
            the winning layout is the task's <b>proof of solvability</b></li>
        <li class="small">Searched K* is an upper bound (coarse family), but
            it includes the strategies agents actually use ·
            gap pruning bounds the sims/task</li>
      </ul>
    </div>
    <div style="flex:1;">
      <img src="data:image/png;base64,{{IMG_TURNLAYOUTS}}"
           alt="geometry-exact corner layout candidates" style="max-height:520px">
      <p class="small" style="margin-top:0.2em;">the k=2 candidate family for one task:
      straight probe, the five natural-corner configs, and the excluded mirrored pair</p>
    </div>
  </div>
  <aside class="notes">Left: the algorithm in compute_turn_k_star /
  _candidate_turn_layouts. Right: geometry-exact renders of the actual
  candidates (make_turn_layouts_fig.py drives the real search code). Panels
  differ by the corner blue's yaw fraction and in/out gaps; higher k adds
  entry blues (which slide the corner along the entry leg) and exit blues.
  The greyed panel is the generator's mirrored pair — physically the most
  robust corner at min-block gaps, but excluded because no planner would
  propose it. If a straight probe topples, the task's true K* is lower
  than the even-L count — exactly the case that broke exact-minimality
  before the search existed.</aside>
</section>

<!-- 12 ─ Turn geometry A/B figure -->
<section>
  <h2>The 45°-block that looks wrong — and works</h2>
  <img src="data:image/png;base64,{{IMG_TURNAB}}" alt="turn yaw A/B + search families" style="max-height:430px">
  <p class="small"><span class="hl">A/B-verified:</span> the natural-looking alignment (panel 2) <b>never
  propagates at min-block gaps</b> (0.098–0.13, both frictions, ±W/2 offsets). At the legacy generator's
  tighter gaps (≲0.09) it works fine — the claim is scoped to the near-reach-limit band. The mirrored yaw
  (panel 1) is load-bearing there: the block is clipped and side-swept into the next one.
  <b>And that is exactly why the K* search excludes it</b> — a corner that only works via an orientation
  no planner would propose must not set the task's budget. Panels 3–4: why K* must <em>search</em> layouts.</p>
  <aside class="notes">Corner candidates replicate the generator's tested
  45°-pair geometry exactly, including the half-width side nudges. The
  straight-line probes exist so K* accounts for corner-cheating solutions.
  A/B: with d1 yaw = rot - td*45 (natural alignment) the chain FAILS
  at gaps 0.098-0.13, frictions 0.1/0.5, side offsets {-W/2, 0, +W/2};
  with the generator's rot + td*45 it topples at gaps up to 0.11.
  SCOPE (added 2026-07-03): the original deck overclaimed "all gaps" — a
  rerun of the A/B on the real corner-family geometry shows natural-yaw
  corners DO propagate at gaps 0.06-0.09 in several friction/offset
  combinations. That is the regime of the LEGACY generator (b2e0f244),
  which used natural yaw (d1_rot = rotation - td*45) plus a
  turn_shift_frac*W offset — its turn sequences cascaded correctly. The
  mirrored yaw arrived in the #40 overhaul and is what makes corners work
  at the wider min-block gaps. The full sim-exact 10-panel candidate grid
  lives at docs/envs/assets/domino_min_block/turn_layouts.png. The
  redirect works by the 45-block being clipped and side-swept into the
  second turn block, not by falling "along the chain". The second turn
  block's yaw is stored 180 deg off its travel — footprint-identical,
  purely cosmetic.</aside>
</section>

<!-- 13 ─ Differentiation filter -->
<section>
  <h2>Not every task separates the models</h2>
  <ul>
    <li><b>Dead band:</b> spans where both frictions need the <em>same</em> count
        (e.g. 0.17–0.23 m: one blue either way) → task can't distinguish calibrated from not</li>
    <li><b>Per-task filter:</b> recompute the <em>believed</em> K* at the planning friction; keep only <b>forced failures</b></li>
    <li>Direction-aware:
      <ul>
        <li>planning &gt; true (over-reach): keep <code>believed &lt; true</code> → forced <b>under-build</b></li>
        <li>planning &lt; true (under-reach): keep <code>true &lt; believed ≤ staged</code> → forced <b>over-build</b></li>
      </ul>
    </li>
    <li>Turns: only <b>long entry legs</b> differentiate (leg scan: 3-vs-2, 4-vs-3, 5-vs-4 cells)</li>
  </ul>
  <aside class="notes">The dead band was found empirically — the first
  functional test showed believed == true on sampled tasks. Instead of
  hand-tuning span bands, the filter simulates the planner's belief per
  task. Every surviving straight task has believed = true − 1. The reverse
  (under-reach) direction exercises the over-build side of the reward;
  weaker learning signal though, since cascades physically succeed.</aside>
</section>

<!-- 14 ─ Sampled tasks: true vs believed -->
<section>
  <h2>Sampled tasks: calibrated vs miscalibrated, side by side</h2>
  <img src="data:image/png;base64,{{IMG_TASKEXAMPLES}}" alt="sampled tasks with true vs believed solutions" style="max-height:520px">
  <p class="small">All five tasks of the live seed-0 test set. Middle: the K*-search's winning layout at the
  <b>true</b> friction (sim-verified ✓). Right: what the <b>µ=0.5</b> model builds — believed chain /
  under-built entry leg — executed at true friction: <span class="bad">dies short ✗</span> on every task.</p>
  <aside class="notes">Generated by make_task_examples_fig.py from the live
  cache: layouts come from the real search code, verdicts from real Push
  rollouts. Straight believed side uses the drift-free canonical span probe;
  the turn believed side rebuilds the same route with one fewer blue
  (each facing its local travel direction), per the leg certificate. If a right-hand panel ever says TOPPLES (leak), the
  differentiation filter has a hole worth investigating.</aside>
</section>

<!-- 14b ─ Heavy-block variant -->
<section>
  <h2>New task type: the heavy-block obstacle</h2>
  <ul>
    <li>A <b>gray, domino-shaped block</b>, true mass <b>1000 kg</b> (untopple-able,
        unmovable); planning sims believe <b>normal domino mass</b>
        (<code>heavy_block_mass</code> override) — a MASS-only mismatch</li>
    <li>Two natural alignments (mixed per <code>turn_ratio</code>):
      <ul>
        <li><b>straight</b>: start → gray → target on one line, all co-facing —
            believed plan chains <em>through</em> the gray for free;
            <span class="good">true solution: a half-circle swerve around it</span></li>
        <li><b>turn</b>: the gray stands exactly where the believed-cheapest L-plan's
            corner blue would go (one blue cheaper than any own-corner plan);
            <span class="good">true solution: skip around with an own corner</span></li>
      </ul></li>
    <li class="bad">Either way the baseline's cheapest validated plan dies against the gray</li>
    <li>Certificate per task (all simulated): believed lure exists at the family
        minimum, lure dead at true physics, true swerve/detour K* within the staged
        blues; <b>budget = the staged blues</b> (binary topple success — K* certifies
        solvability only; corner minima are solver-history sensitive at the margin)</li>
    <li class="small">No friction mismatch here: corners never propagate at µ=0.5,
        which would kill the turn lure — this env isolates the mass dimension</li>
  </ul>
  <aside class="notes">Single flag: domino_heavy_block_tasks. Differentiates
  on a discrete per-object property (mass heterogeneity) instead of the
  global friction — uniform mass is unidentifiable and irrelevant, but the
  striker/struck mass RATIO is exactly what the chain outcome exposes. The
  turn lure is structural: the gray replaces the corner of a minimum-cost
  believed layout, so it is strictly one blue cheaper than every own-corner
  alternative. Swerve physics: heading profile phi*sin(2*pi*t), knocks stay
  within the ~33-degree propagation tolerance at phi 30-40 degrees.</aside>
</section>

<!-- 14c ─ Heavy-block sampled tasks -->
<section>
  <h2>Heavy-block tasks: calibrated vs miscalibrated, side by side</h2>
  <img src="data:image/png;base64,{{IMG_HEAVYEXAMPLES}}" alt="heavy-block tasks with believed dogleg vs detour solutions" style="max-height:520px">
  <p class="small">The live seed-0 heavy-block test set. Top: staged init — the <b>gray block</b>
  (1000 kg) sits dead ahead on the line (straight) or at the L's natural corner (turn).
  Middle: the calibrated solution — half-circle swerve / skip-around detour — sim-verified
  at the true physics: <span class="good">topples ✓</span>. Bottom: the believed
  (<b>normal-mass</b>) plan through the gray, executed at the true physics:
  <span class="bad">dies at the gray block ✗</span>.</p>
  <aside class="notes">Generated by make_heavy_task_examples_fig.py from the
  live cache: the believed row re-derives the cheapest through-gray plan the
  believed physics accepts (the same families the generation certificates
  scan), then executes it at the true physics; the calibrated row is the
  swerve/detour search's winning layout, re-verified. If a bottom panel ever
  says TOPPLES (leak), the true-dead certificate has a hole worth
  investigating.</aside>
</section>

<!-- 15 ─ Engineering -->
<section>
  <h2>Engineering that mattered</h2>
  <ul>
    <li><b>Friction roles:</b> eval env = true friction; planning sims
        (<code>skip_process_dynamics=True</code>) = planning friction; an oracle flag grants ground truth</li>
    <li><b>Staging capacity:</b> the staging grid fits ≤ 4 blues (gripper clearance)
        → enforce K* ≤ 3 so a spare blue always exists (over-build stays penalized)</li>
    <li><b>Task cache:</b> keyed on (config flags, seed, <em>source-code digest</em>) —
        43 s → 0.01 s per reload; auto-invalidates on any code/config change;
        <b>all arms share identical tasks</b></li>
    <li><b>Determinism:</b> physics params are never altered for speed — K* is <em>defined</em> by eval physics</li>
  </ul>
  <aside class="notes">The friction role split needs no approach-side code:
  every planning env in the codebase already self-identifies via
  skip_process_dynamics=True. The staging cap silently zeroed task
  generation when we tried 6 blues — worth knowing. Cache lives in
  saved_datasets/domino_min_block_tasks.</aside>
</section>

<!-- 16 ─ Matrix -->
<section>
  <h2>The experiment matrix</h2>
  <table>
    <tr><th>arm</th><th>planner's friction</th><th>expected</th><th>establishes</th></tr>
    <tr><td>no-learning baseline<br><span class="small">agent_base_sim_no_learning</span></td>
        <td>0.5 (wrong)</td><td class="bad">~0/5</td>
        <td>miscalibration ⇒ task failure (unrecoverable)</td></tr>
    <tr><td>oracle: GT hybrid sim + params<br><span class="small">agent_oracle_hybrid_sim_no_demo</span></td>
        <td>0.1 (true)</td><td class="good">~5/5</td>
        <td>tasks solvable; LLM commits to exactly-K* plans</td></tr>
    <tr><td>ours + friction sysID<br><span class="small">(next build)</span></td>
        <td>0.5 → <b>learns</b> → ~0.1</td><td class="hl">fail → learn → succeed</td>
        <td>calibration is learnable from interaction</td></tr>
  </table>
  <p class="small">Same pipeline family (agent_sim_learning) for oracle and ours — the contrast isolates <em>what is learned</em>.</p>
  <aside class="notes">The oracle doubles as the critical control: if IT
  fails, the problem is planning/minimality, not the model — fix tasks
  before blaming sysID. Baseline failure is structural: the differentiation
  filter guarantees its minimal plan under-builds, and the fallen start
  can't be re-pushed.</aside>
</section>

<!-- 17 ─ Status & discussion -->
<section>
  <h2>Status &amp; discussion</h2>
  <ul>
    <li><b>Built + verified in sim:</b> two-sided reward, straight + 90°-turn generator,
        searched K*, direction-aware filters, task cache, baseline &amp; oracle configs</li>
    <li><b>Next:</b> the friction sysID fit — LM over interaction roll/z trajectories,
        injected into the planning sim each online cycle</li>
  </ul>
  <p><b>Open questions for this group</b></p>
  <ul>
    <li>Searched K* is an upper bound over the layout family — tight enough?</li>
    <li>Reverse (over-build) condition: report as a second axis, or appendix?</li>
    <li>Calibration numbers are single-seed — how many seeds for the paper?</li>
  </ul>
  <aside class="notes">Everything is on branch test-sysid-utility-domino,
  uncommitted. Launch command: python scripts/local/launch_simp.py -c
  predicatorv3/agents.yaml. Recommend one seed of each arm first; each solve
  session costs LLM budget.</aside>
</section>

</div></div>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/dist/reveal.js"></script>
<script src="https://cdn.jsdelivr.net/npm/reveal.js@4.6.1/plugin/notes/notes.js"></script>
<script>
  Reveal.initialize({ hash: true, slideNumber: true, plugins: [ RevealNotes ] });
</script>
</body>
</html>
"""

out = HTML.replace("{{IMG_PHASE1}}", b64("phase1_gap.png")) \
          .replace("{{IMG_ANATOMY}}", b64("task_anatomy.png")) \
          .replace("{{IMG_TURNAB}}", b64("turn_ab.png")) \
          .replace("{{IMG_TURNLAYOUTS}}", b64("turn_layouts.png")) \
          .replace("{{IMG_TASKEXAMPLES}}", b64("task_examples.png")) \
          .replace("{{IMG_HEAVYEXAMPLES}}", b64("heavy_task_examples.png"))
dest = HERE.parent.parent / "slides" / "domino_min_block_task_gen_slides.html"
dest.write_text(out)
print("wrote", dest, f"({len(out)//1024} KB)")
