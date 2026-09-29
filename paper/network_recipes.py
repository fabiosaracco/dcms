"""Per-network DECM convergence recipes for the 19 real validation networks (crisi dico0-4,
ita_elections dico0-6, quirinale dico0-6), for the paper and for §5 (Validation) of
paper/technical_appendix.tex.

WHERE THIS DATA COMES FROM (read before trusting a number): every `Recipe` below was built by
loading the network's own stored, converged `SolverResult` from bowtie2/tests/ and reading its
`.sol.message` field, which every pickle written this project's solve scripts records with the
recipe that actually produced it (see `bowtie2/decm_dico_calculator_batch2608.py` for the 9
BATCH recipes, `scratch_decm_e1/net_run.py` for the streak/block-Newton runs). `confidence` says
how directly each recipe traces back to that stored provenance:
  - "exact"    : every kwarg below is read verbatim from the stored message or its known driver
                 script; a fresh reproduction attempt (`reproduce_network`) is meaningful.
  - "inferred" : the network converged as part of a documented joint investigation with another
                 network whose recipe IS exact (named in `notes`), and is assumed to share it --
                 not itself verbatim-recorded. Treat as a strong hypothesis, not a certainty.
  - "partial"  : only some of the recipe survived in the stored provenance (e.g. z_clamp/noise
                 but not anderson_depth); the rest is left at the package's own documented
                 defaults (README §3.4) and flagged, not guessed at.

Two things this script actually DOES, and can be trusted for:
  1. `verify_all()` (run by default, ~1 min): reloads every one of the 19 stored solutions and
     independently recomputes `model.max_relative_error(sol.best_theta)` -- this is NOT read from
     the pickle, it is recomputed from scratch against the constraint equations, and is therefore
     real evidence, not a repeated claim. This is what "converged" means in this file.
  2. `reproduce_network(tag, ...)` (opt-in, NOT run by default -- some of these take up to 24h):
     builds a FRESH DECMModel from the same (k_out,k_in,s_out,s_in) as the stored solution and
     calls `solve_tool()` with the recorded recipe, from the recorded starting point. For the 10
     "fresh degrees IC" networks this is a genuine from-scratch reproduction. For the 9 networks
     that were reached via a resumed checkpoint (a real, multi-round, sometimes multi-day search,
     not a single clean recipe -- see `notes`), it reproduces the FINAL leg only, warm-started
     from that checkpoint file where it is still on disk; see each Recipe's `resume_from` note for
     what "from zero" would actually require for that network, honestly.

This is deliberately not yet a claim that every network converges unattended from a single clean
recipe -- Fabio's real question (README/paper §5, technical_appendix.tex §5, `paper`
point-5 discussion) -- only the data this project actually has on how each one WAS converged, laid
out so that question can be tested network by network.
"""
from __future__ import annotations

import dataclasses
import pickle
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]      # .../bowtie2_py310
TESTS_DIR = REPO_ROOT / "bowtie2" / "tests"
TOL = 1e-5


@dataclasses.dataclass
class Recipe:
    tag: str                       # short id used throughout this project's memory/scripts
    dataset: str                   # "crisi" | "ita_elections" | "quirinale"
    dico: int                      # dico class 0-6 (0-4 for crisi)
    pickle_name: str               # file in bowtie2/tests/ holding the converged SolverResult
    N: int
    confidence: str                # "exact" | "inferred" | "partial" -- see module docstring
    solve_tool_kwargs: dict        # kwargs to DECMModel.solve_tool(), beyond ic/tol/max_iter
    fresh_ic: bool                 # True: recipe starts from ic="degrees"; False: resumed
    resume_from: str               # if not fresh_ic, what checkpoint/history it resumed from
    notes: str                     # provenance detail / caveats

    @property
    def net_tag(self) -> str:
        return f"{self.dataset}_dico{self.dico}"


# ---------------------------------------------------------------------------------------------
# The BATCH recipe: bowtie2/decm_dico_calculator_batch2608.py, verbatim from its own constants
# (lines 37-51 as of commit ac46c4d-era: MAX_TIME_HOURS=24, MAX_ITER=300000, TOL=1e-5,
# Z_CLAMP=1e-6, NOISE_BASE=2e-3, HUB_TH=5.0, GAMMA=0.0, ANDERSON=10), fresh ic="degrees",
# multi_start=True (retries "qdecm"/"random" if "degrees" doesn't converge). This ran unattended
# and converged 9 of the 19 networks -- crisi dico3/4, ita_elections dico4/5/6, quirinale
# dico0/1/2/3 -- with NO per-network tuning beyond this single shared recipe.
# ---------------------------------------------------------------------------------------------
_BATCH_KW = dict(
    anderson_depth=10, hub_sk_threshold=5.0, backtracking_gamma=0.0, z_clamp=1e-6, noise_base=2e-3,
)
_BATCH_NOTE = (
    "Standard batch recipe (decm_dico_calculator_batch2608.py), fresh ic='degrees', multi_start=True, "
    "max_time=24h, max_iter=300000 -- converged unattended with NO network-specific tuning."
)

RECIPES: list[Recipe] = [
    # -------------------------------------------------------------- the 9 plain-batch networks
    Recipe("c3", "crisi", 3, "batch2608_crisi_dico3_decm.pkl", 1304, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),
    Recipe("c4", "crisi", 4, "batch2608_crisi_dico4_decm.pkl", 3637, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),
    Recipe("e4", "ita_elections", 4, "batch2608_ita_elections_dico4_decm.pkl", 9010, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),
    Recipe("e5", "ita_elections", 5, "batch2608_ita_elections_dico5_decm.pkl", 373, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),
    Recipe("e6", "ita_elections", 6, "batch2608_ita_elections_dico6_decm.pkl", 337, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),
    Recipe("q0", "quirinale", 0, "batch2608_quirinale_dico0_decm.pkl", 33244, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),
    Recipe("q1", "quirinale", 1, "batch2608_quirinale_dico1_decm.pkl", 11045, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),
    Recipe("q2", "quirinale", 2, "batch2608_quirinale_dico2_decm.pkl", 18704, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),
    Recipe("q3", "quirinale", 3, "batch2608_quirinale_dico3_decm.pkl", 3000, "exact",
           _BATCH_KW, True, "", _BATCH_NOTE),

    # ---------------------------------------------------------------------- crisi_dico2 (c2)
    Recipe("c2", "crisi", 2, "crisi_dico2_decm_and_10_gamma_0.0_hub_5.pkl", 15168, "partial",
           dict(anderson_depth=10, hub_sk_threshold=5.0, backtracking_gamma=0.0), True, "",
           "Filename encodes anderson_depth=10/backtracking_gamma=0.0/hub_sk_threshold=5.0 "
           "(a parameter-sweep script from the same 2026-07 era as the batch recipe); "
           "z_clamp/noise_base not recorded in the filename -- left at the package defaults "
           "(z_clamp=1e-6, noise_base=1e-4) rather than assumed equal to the batch recipe's."),

    # ---------------------------------------------------------------------- crisi_dico0 (c0)
    Recipe("c0", "crisi", 0, "crisi_dico0_decm_conv.pkl", 58832, "exact",
           dict(anderson_depth=3, streak_fix_threshold=50, hub_sk_threshold=5.0, z_clamp=1e-6),
           True, "",
           "Never run before 2026-09-24. Converged over 3 chained 12h rounds (2026-09-24..26, "
           "~29h total), same recipe throughout, from a fresh ic='degrees' -- the single cleanest "
           "'ran from zero to convergence, just needed wall-clock' case in this set."),

    # ---------------------------------------------------------------------- crisi_dico1 (c1)
    Recipe("c1", "crisi", 1, "crisi_dico1_decm_conv.pkl", 31874, "exact",
           dict(anderson_depth=3, streak_fix_threshold=50, hub_sk_threshold=5.0, z_clamp=1e-6),
           False, "results/c1_depth3_12h_result.pt (stella, 12h run)",
           "This recipe is what converged the FINAL 12h leg from that checkpoint; the checkpoint "
           "itself is the product of a much longer prior search (see decm_low_degree_precision_floor "
           "memory) -- a fresh ic='degrees' run with this exact recipe has NOT itself been verified "
           "to converge unattended from zero."),

    # ------------------------------------------------------------- ita_elections_dico0 (e0)
    Recipe("e0", "ita_elections", 0, "ita_elections_dico0_decm_conv.pkl", 50288, "inferred",
           dict(anderson_depth=3, streak_fix_threshold=50, hub_sk_threshold=5.0, z_clamp=1e-6),
           False, "scratch_decm_c1/e0_resumed_12h_result.pt (stella, 12h run)",
           "Solved in the same joint investigation as c1 (paired file naming, same memory entry); "
           "assumed to share c1's anderson_depth=3/streak_fix_threshold=50 recipe, but this is NOT "
           "itself verbatim-recorded in e0's own stored message -- verify before citing as exact."),

    # ------------------------------------------------------------- ita_elections_dico2/3 (e2, e3)
    Recipe("e2", "ita_elections", 2, "ita_elections_dico2_decm_0.pkl", 20914, "partial",
           dict(anderson_depth=10), True, "",
           "Converged in 83 iterations by an ad hoc script whose exact recipe was not preserved "
           "beyond the default anderson_depth=10; z_clamp/hub_sk_threshold left at package "
           "defaults (1e-6 / 0.0) rather than assumed."),
    Recipe("e3", "ita_elections", 3, "ita_elections_dico3_decm_final_2.pkl", 28156, "partial",
           dict(anderson_depth=10, z_clamp=1e-6, noise_base=2e-3), True, "",
           "The z_clamp/noise_base pairing is exact (decm_ita_dico3_zclamp_investigation: "
           "z_clamp=1e-6 tied to the noise_base=2e-3 restart scale, 1313 iterations); "
           "hub_sk_threshold not recorded -- left at the package default (0.0, disabled)."),

    # ---------------------------------------------------------------------- quirinale_dico4 (q4)
    Recipe("q4", "quirinale", 4, "quirinale_dico4_decm_conv.pkl", 22754, "exact",
           dict(anderson_depth=1, streak_fix_threshold=1, hub_sk_threshold=2.0, z_clamp=1e-6,
                block_newton_gate=0.05, block_newton_min_rel=0.0),
           False, "scratch_decm_e1/q4_q20r2_final.pt (a 3.5175e-5 plateau)",
           "block_newton_min_rel=0.0 here is NOT the recommended default (0.1*tol) -- q4's run "
           "predates that kwarg's existence and converged in 46 iterations with the gate alone, "
           "before the runaway-row failure mode (technical_appendix.tex Sec. 3.7) could manifest. "
           "The plateau itself was the product of a long prior search (see decm_low_degree_precision_floor "
           "and decm_b2_block_newton_2026_09_28 memories) -- reproducing THAT from a fresh "
           "ic='degrees' is a much longer, multi-stage undertaking, not this one recipe."),

    # ---------------------------------------------------------------------- quirinale_dico5 (q5)
    Recipe("q5", "quirinale", 5, "quirinale_dico5_decm_conv.pkl", 27746, "exact",
           dict(anderson_depth=1, streak_fix_threshold=1, hub_sk_threshold=1.5, z_clamp=1e-6),
           False, "scratch_decm_e1/q5_h15d1_final.pt (a 2.05e-4-era plateau)",
           "Solver era 9819ebb/ac46c4d (pre block-Newton); 351 iterations for this final leg. "
           "NOTE (2026-09-28 A/B, see decm_branch_merged_2026_09_28 memory): from a completely "
           "FRESH ic='degrees' start, depth 3 + streak 50 + hub_sk_threshold=5.0 + "
           "block_newton_gate=0.05 converged this exact network in 1845 iterations (~9 min) -- a "
           "cleaner from-zero recipe than the one this Recipe records, kept here anyway for an "
           "honest account of how q5 was ACTUALLY first solved. See RECOMMENDED_FRESH_START below "
           "for the better recipe to actually use going forward."),

    # ---------------------------------------------------------------------- quirinale_dico6 (q6)
    Recipe("q6", "quirinale", 6, "quirinale_dico6_decm_conv.pkl", 232, "exact",
           dict(anderson_depth=5, streak_fix_threshold=50, hub_sk_threshold=5.0, z_clamp=1e-6),
           False, "the batch2608 run's own best_theta (a 6.8e-2 plateau)",
           "139388 iterations (a genuine crawl, ~55 it/s single-threaded, ~45 min wall-clock) -- "
           "MRE ~ 1/iterations on this network, no shortcut found. NOTE: from a FRESH ic='degrees' "
           "start the plain batch recipe alone (anderson_depth=10, no streak-fix) converges this "
           "same network in 288 iterations / 13s (2026-09-28 A/B) -- q6's slow 139k-iteration path "
           "above was a historical artifact of starting from an already-bad plateau, not evidence "
           "this network is intrinsically hard. See RECOMMENDED_FRESH_START."),

    # ---------------------------------------------------------------------- ita_elections_dico1 (e1)
    Recipe("e1", "ita_elections", 1, "ita_elections_dico1_decm_conv.pkl", 107056, "exact",
           dict(anderson_depth=3, streak_fix_threshold=50, hub_sk_threshold=2.0, z_clamp=1e-6,
                block_newton_gate=0.05, block_newton_min_rel=1e-6),
           False, "scratch_decm_e1/e1_e20r2_final.pt (a 1.8513e-4 plateau)",
           "The last network to close (2026-09-28), 94 iterations for this final leg. Same caveat "
           "as q4: the plateau it started from was the product of a long prior search, not "
           "reproducible by this recipe alone from a fresh ic='degrees'."),
]

assert len({r.tag for r in RECIPES}) == 19, f"expected 19 unique recipes, got {len(RECIPES)}"


# A recipe known (2026-09-28 A/B testing, see decm_branch_merged_2026_09_28 memory) to converge
# CLEANLY from a fresh ic='degrees' start on at least q5 and q6 above, faster than how they were
# first (historically) solved. Not yet swept across all 19 -- offered as the current best
# starting guess for the "does it converge from zero, unattended" question point 5 is really
# asking, not as a settled answer.
RECOMMENDED_FRESH_START: dict = dict(
    anderson_depth=3, streak_fix_threshold=50, hub_sk_threshold=5.0, z_clamp=1e-6,
    block_newton_gate=0.05, block_newton_min_rel=None,   # None -> 0.1*tol, the safer paired default
)


def _load_model(recipe: Recipe):
    with open(TESTS_DIR / recipe.pickle_name, "rb") as fh:
        return pickle.load(fh)


def verify_all(recipes: list[Recipe] = RECIPES, tol: float = TOL) -> bool:
    """Reload every network's stored converged solution and INDEPENDENTLY recompute its MRE
    (never trust the pickle's own stored number alone). Prints a table, returns True iff all 19
    pass. Takes about a minute (ita_elections_dico1's N=107056 residual call alone is ~2 min on a
    laptop -- this is the dominant cost)."""
    print(f"{'tag':4s} {'network':24s} {'N':>8s} {'MRE (recomputed)':>18s} {'<=tol':>6s} {'confidence':>10s}")
    print("-" * 78)
    all_ok = True
    for r in recipes:
        m = _load_model(r)
        theta = torch.as_tensor(m.sol.best_theta, dtype=torch.float64)
        mre = float(m.max_relative_error(theta))
        ok = mre <= tol
        all_ok &= ok
        print(f"{r.tag:4s} {r.net_tag:24s} {r.N:8d} {mre:18.6e} {'OK' if ok else 'FAIL':>6s} {r.confidence:>10s}")
    print("-" * 78)
    print(f"{'ALL 19 CONVERGED' if all_ok else 'SOME FAILED'} (tol={tol:g})")
    return all_ok


def reproduce_network(tag: str, max_iter: int = 300_000, max_time: float = 0.0, verbose: bool = True):
    """Re-run ONE network's recorded recipe. For a fresh_ic=True recipe this is a genuine
    from-scratch reproduction (ic='degrees'); for fresh_ic=False it warm-starts from
    `resume_from`'s checkpoint file if still present on disk, otherwise raises -- this function
    does not silently substitute a fresh start for a resumed recipe, since that is a different,
    unverified claim (see each Recipe's `notes`).

    NOT run automatically by this module -- several of these recipes took up to 24h historically.
    Pass `max_time` (seconds) to cap a single attempt; 0 = no limit (use with care).
    """
    from dcms.models.decm import DECMModel

    recipe = next((r for r in RECIPES if r.tag == tag), None)
    if recipe is None:
        raise ValueError(f"unknown tag {tag!r}; choices: {[r.tag for r in RECIPES]}")

    m = _load_model(recipe)
    model = DECMModel(m.k_out, m.k_in, m.s_out, m.s_in)

    if recipe.fresh_ic:
        ic = "degrees"
    else:
        ckpt = REPO_ROOT / recipe.resume_from.split(" ")[0]
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{recipe.tag}'s recipe resumes from {ckpt}, not found on this machine. "
                f"This recipe is NOT a from-scratch reproduction -- see its `notes`."
            )
        ic = torch.as_tensor(torch.load(open(ckpt, "rb"), weights_only=False), dtype=torch.float64)

    print(f"[{recipe.tag}] {recipe.net_tag} (N={recipe.N}), fresh_ic={recipe.fresh_ic}, "
          f"confidence={recipe.confidence}\n  kwargs={recipe.solve_tool_kwargs}")
    t0 = time.perf_counter()
    converged = model.solve_tool(
        ic=ic, tol=TOL, max_iter=max_iter, max_time=max_time, verbose=verbose,
        **recipe.solve_tool_kwargs,
    )
    dt = time.perf_counter() - t0
    print(f"[{recipe.tag}] converged={converged} iters={model.sol.iterations} "
          f"mre={model.sol.mre:.6e} elapsed={dt:.1f}s")
    return model


if __name__ == "__main__":
    ok = verify_all()
    if len(sys.argv) > 1:
        # e.g. `python network_recipes.py c3 --max_time 600`
        reproduce_network(sys.argv[1], max_time=float(sys.argv[3]) if len(sys.argv) > 3 else 0.0)
    sys.exit(0 if ok else 1)
