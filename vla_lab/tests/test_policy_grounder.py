"""The trained policy acting as the robot's grounding channel.

These tests exist because the closed-loop evaluation has a failure mode the offline metrics
cannot see: a checkpoint that scores well on held-out dialogues but, plugged into the session,
either abstains on everything (starving the estimand) or never abstains at all (beating the
lexical reference by guessing where the reference declines). Both would still produce a
perfectly ordinary-looking table.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

try:
    import torch

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

from vla_lab.supervisory import STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED
from vla_lab.supervisory.contract import Contract
from vla_lab.tests import approx


def _skip() -> bool:
    if not HAVE_TORCH:
        print("      (torch unavailable; skipped)")
        return True
    return False


def _checkpoint(tmp: Path, context_mode: str = "film") -> Path:
    """A real (untrained) CarryoverVLA saved in the exact format the trainer writes."""
    from dataclasses import asdict

    from vla_lab.dataset import TinyTokenizer
    from vla_lab.policy.registry import MODEL_CARDS, build_model

    tok = TinyTokenizer.build_from_corpus(
        ["clear the blocker first", "just go straight for it", "move the box out of the way"], max_len=24
    )
    model = build_model("tiny", context_mode=context_mode, vocab_size=tok.vocab_size)
    path = tmp / "best.pt"
    torch.save({"model": model.state_dict(), "config": model.cfg.to_dict(), "tokenizer": asdict(tok),
                "manifest": {"context_mode": context_mode, "context_style": "compact",
                             "card": MODEL_CARDS["tiny"].to_dict(),
                             "data": {"vocab_size": int(tok.vocab_size)}}}, path)
    return path


def test_grounds_through_the_protocol():
    """A checkpoint satisfies the Grounder protocol and returns only legal labels."""
    if _skip():
        return
    from vla_lab.policy.grounder import PolicyGrounder
    from vla_lab.supervisory.apparatus.base import Grounder

    contract = Contract()
    with tempfile.TemporaryDirectory() as td:
        g = PolicyGrounder(_checkpoint(Path(td)), axis=contract.axis, read="said", device="cpu")
        assert isinstance(g, Grounder), "PolicyGrounder must satisfy the runtime protocol"
        for scene in contract.grid.scenes[:5]:
            label = g.ground("clear the blocker first please", scene)
            assert label in (STRATEGY_A, STRATEGY_B, STRATEGY_UNRESOLVED), f"illegal label {label!r}"


def test_abstains_below_threshold_and_never_above():
    """The confidence gate is what makes the comparison against the lexical channel fair.

    An untrained model is near-chance, so a threshold above 0.5 must abstain on essentially
    everything and a threshold of 0.0 must abstain on nothing. If either direction fails the
    gate is not wired to the probabilities at all.
    """
    if _skip():
        return
    from vla_lab.policy.grounder import PolicyGrounder

    contract = Contract()
    scenes = contract.grid.scenes[:8]
    with tempfile.TemporaryDirectory() as td:
        ckpt = _checkpoint(Path(td))
        strict = PolicyGrounder(ckpt, axis=contract.axis, min_confidence=0.999, device="cpu")
        loose = PolicyGrounder(ckpt, axis=contract.axis, min_confidence=0.0, device="cpu")
        for s in scenes:
            strict.ground("clear it first", s)
            loose.ground("clear it first", s)
        assert strict.report()["abstain_rate"] > 0.9, strict.report()
        assert loose.report()["abstain_rate"] == 0.0, loose.report()


def test_context_changes_the_reading():
    """Injected carryover context must reach the network.

    A grounder that ignored ``set_context`` would make the whole closed-loop de-biasing result
    vacuous -- it would be reading the utterance and nothing else -- and nothing else in the
    pipeline would notice, because the labels it returns would still look entirely reasonable.

    ``token`` mode is used because ``film`` is deliberately identity-initialised (zeroed final
    layer, so an untrained FiLM head cannot scramble a pretrained backbone on step zero), which
    means an untrained FiLM model is *supposed* to be context-invariant. The FiLM path is
    checked separately below, precisely because that init hides a wiring bug.
    """
    if _skip():
        return
    from vla_lab.policy.context import CarryoverContext
    from vla_lab.policy.grounder import PolicyGrounder

    contract = Contract()
    scene = contract.grid.scenes[len(contract.grid.scenes) // 2]
    with tempfile.TemporaryDirectory() as td:
        g = PolicyGrounder(_checkpoint(Path(td), context_mode="token"), axis=contract.axis,
                           read="unprompted", min_confidence=0.0, device="cpu")

        def probs(kappa: float):
            g.set_context(CarryoverContext(kappa=kappa, kappa_sd=0.2, lambda_hat=0.6,
                                           slots_since_coach=1, recent=(("A", 1, True),),
                                           scene_c=float(scene.c)))
            g.ground("clear the blocker first", scene)
            return g.last_probs

        a, b = probs(+1.5), probs(-1.5)
        assert abs(a[0] - b[0]) > 1e-6, f"context did not reach the network: {a} vs {b}"


def test_film_path_is_wired_despite_identity_init():
    """FiLM is zero-initialised on purpose, so 'no effect' cannot be read as 'connected'.

    Perturbing the FiLM projection has to change the reading. If it does not, the context is
    not reaching the heads at all and the identity init is quietly covering for it.
    """
    if _skip():
        return
    from vla_lab.policy.context import CarryoverContext
    from vla_lab.policy.grounder import PolicyGrounder

    contract = Contract()
    scene = contract.grid.scenes[len(contract.grid.scenes) // 2]
    ctx = CarryoverContext(kappa=1.5, kappa_sd=0.2, lambda_hat=0.6, slots_since_coach=1,
                           recent=(("A", 1, True),), scene_c=float(scene.c))
    with tempfile.TemporaryDirectory() as td:
        g = PolicyGrounder(_checkpoint(Path(td), context_mode="film"), axis=contract.axis,
                           read="unprompted", min_confidence=0.0, device="cpu")
        g.set_context(ctx)
        g.ground("clear the blocker first", scene)
        before = g.last_probs
        with torch.no_grad():
            torch.nn.init.normal_(g.model.context.net[-1].weight, std=0.05)
        g.ground("clear the blocker first", scene)
        assert abs(g.last_probs[0] - before[0]) > 1e-6, "FiLM context is not connected to the heads"


def test_report_is_well_formed():
    if _skip():
        return
    from vla_lab.policy.grounder import PolicyGrounder

    contract = Contract()
    with tempfile.TemporaryDirectory() as td:
        g = PolicyGrounder(_checkpoint(Path(td)), axis=contract.axis, min_confidence=0.0, device="cpu")
        g.ground("clear the blocker out of the way first", contract.grid.scenes[0])
        r = g.report()
        for k in ("grounder", "read", "calls", "abstain_rate", "agreement_with_lexical", "context_mode"):
            assert k in r, f"report missing {k}"
        assert r["calls"] == 1
        assert 0.0 <= r["agreement_with_lexical"] <= 1.0


def test_session_supplies_context_to_a_learned_grounder():
    """The session loop must call ``set_context`` -- the belief has to flow to the model."""
    from vla_lab.supervisory.contract import Contract
    from vla_lab.supervisory.protocol import build_protocol
    from vla_lab.supervisory.scheduler import CONDITION_CARRYOVER_AWARE, build_scheduler
    from vla_lab.supervisory.session import run_block
    from vla_lab.supervisory.supervisor import SimulatedSupervisor, draw_supervisor
    from vla_lab.supervisory.apparatus import (LexicalGrounder, SimulatedSupervisorChannel,
                                               SurrogateApparatus)
    import random

    contract = Contract()

    class Spy(LexicalGrounder):
        name = "spy"

        def __init__(self, axis):
            super().__init__(axis)
            self.contexts = []

        def set_context(self, ctx):
            self.contexts.append(ctx)

    rng = random.Random(3)
    params = draw_supervisor(rng, None, supervisor_id="S000")
    sup = SimulatedSupervisor(params, axis=contract.axis, cfg=contract.carryover, seed=5)
    proto = build_protocol(supervisor_id="S000", contract=contract, seed=5,
                           conditions=[CONDITION_CARRYOVER_AWARE], order_index=0)
    block = proto.blocks[0]
    sch = build_scheduler(CONDITION_CARRYOVER_AWARE, contract.grid, carryover_cfg=contract.carryover,
                          delta_model=contract.delta_model(), seed=5)
    spy = Spy(contract.axis)
    run_block(contract=contract, block=block, scheduler=sch, channel=SimulatedSupervisorChannel(sup),
              apparatus=SurrogateApparatus(contract.grid, seed=5), grounder=spy, seed=5)
    assert spy.contexts, "run_block never handed the grounder a context"
    ctx = spy.contexts[-1]
    assert hasattr(ctx, "kappa") and hasattr(ctx, "slots_since_coach")
    assert -8.0 < float(ctx.kappa) < 8.0, f"implausible kappa handed to the grounder: {ctx.kappa}"



def test_the_ask_decision_comes_from_the_belief_module_never_from_the_learned_gate():
    """P2-2. The learned ask gate is a functional of the quantity under estimation and cannot be
    identified from data in which that quantity is unknown; the deployed system therefore takes
    the counter-proposal decision from the explicit belief module. Concretely: a grounder that
    screams "ask!" on every call must not change a single action the session takes."""
    import random

    from vla_lab.supervisory.apparatus import LexicalGrounder, SimulatedSupervisorChannel, SurrogateApparatus
    from vla_lab.supervisory.contract import Contract
    from vla_lab.supervisory.protocol import build_protocol
    from vla_lab.supervisory.scheduler import POLICY_RECOMMENDED, build_scheduler
    from vla_lab.supervisory.session import run_block
    from vla_lab.supervisory.supervisor import SimulatedSupervisor, draw_supervisor

    contract = Contract()

    class LoudGate(LexicalGrounder):
        """Grounds like the lexical channel and carries an ask gate pinned to certainty."""

        name = "loud-gate"
        ask_logit = 1e9
        ask_calls = 0

        def set_context(self, ctx):
            self.last_context = ctx

        def should_ask(self, *a, **k):                      # nothing in the session may call this
            LoudGate.ask_calls += 1
            return True

    def actions(grounder):
        rng = random.Random(11)
        params = draw_supervisor(rng, None, supervisor_id="S011")
        proto = build_protocol(supervisor_id="S011", contract=contract, seed=11,
                               conditions=[POLICY_RECOMMENDED], order_index=0)
        block = proto.condition_blocks()[0]
        sup = SimulatedSupervisor(params, axis=contract.axis, cfg=contract.carryover, seed=11)
        sch = build_scheduler(POLICY_RECOMMENDED, contract.grid, carryover_cfg=contract.carryover,
                              delta_model=contract.delta_model(), seed=11)
        res = run_block(contract=contract, block=block, scheduler=sch, channel=SimulatedSupervisorChannel(sup),
                        apparatus=SurrogateApparatus(contract.grid, seed=11), grounder=grounder, seed=11)
        return [r["action"] for r in res.records], [r["rationale"].get("reason") for r in res.records]

    a_lex, why_lex = actions(LexicalGrounder(contract.axis))
    a_loud, why_loud = actions(LoudGate(contract.axis))
    assert a_lex == a_loud, "the grounder's opinion about asking leaked into the action sequence"
    assert LoudGate.ask_calls == 0, "the session consulted the grounder's gate"
    assert any("counter" in str(w).lower() or "posterior" in str(w).lower() or "washout" in str(w).lower()
               for w in why_loud if w), "every ask/wait decision must carry a belief-module rationale"
