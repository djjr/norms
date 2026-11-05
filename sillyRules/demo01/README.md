# Silly Rules

An inspectable learning simulation of Köster, Hadfield-Menell, Everett, Weidinger, Hadfield & Leibo (2022), "Spurious normativity enhances learning of compliance and enforcement behavior in artificial agents," *PNAS* 119(3), e2106028118. Built for HMIA.

Eight agents forage for berries. Pink berries are poisonous, but the harm arrives 100 steps after eating, too late for an agent to tell which berry caused it. A taboo marks the violator (visible only to others), and others are rewarded for punishing marked agents. The question is whether adding a second, harmless taboo (teal) helps or hurts the group.

Open `index.html` in a browser. It is one self-contained file with no dependencies, so it also works on GitHub Pages.

## What the model is (and isn't)

This is not deep RL. Each agent learns about ten numbers, and the page lets you inspect them:

| Learned quantity | Scope | Role |
|---|---|---|
| `Q[c]`, how good berry color `c` is | one per color | **compliance**, which is norm-specific |
| `Vm` / `Vu`, value of chasing a marked / innocent agent | shared | willingness to enforce, and discrimination |
| `practice` → `skill` (S-shaped curve) | shared across all norms by default | **enforcement**, which transfers across norms |

That table is the paper's key asymmetry. Enforcement (spot the mark, chase, zap) is the same skill whatever rule was broken. Compliance has to be learned berry by berry. More taboos mean more enforcement practice, which makes punishment of the *important* violation faster and surer, which makes it easier to learn from.

**Credit assignment** is modeled explicitly. Punishment and "feeling sick" both blame recent meals, with blame decaying exponentially in the delay (`creditDecay`). Fast punishment lands on the right berry. Slow punishment, or a 100-step poison delay, gets spread across innocent berries.

## Repo layout

```
index.html                 built page (do not edit by hand)
src/engine.js              the model: world, agents, learning rules. Runs in browser and Node.
src/template.html          the page: canvas worlds, SVG charts, controls, inspector
build.py                   pastes engine.js into template.html -> index.html
scripts/run-conditions.js  headless run of all three conditions, binned averages
scripts/compare-silly-vs-important.js   silly minus important on group return, with t-stats
```

Edit `src/`, then run `python3 build.py`.

## Running experiments headless

```bash
node scripts/run-conditions.js 40 12                          # 40 episodes, 12 populations per rule
KEYS=poisoned,skill node scripts/run-conditions.js 40 16 '{"transferSkill":false}'
EPS=40 SEEDS=20 BASE=5000 node scripts/compare-silly-vs-important.js '[{}, {"zapCost":10}]'
```

Any engine parameter (see `DEFAULTS` in `src/engine.js`) can be overridden with the JSON argument.

## What it shows, and what it doesn't

Robust across seeds (default settings):

- **Sequence.** Punishment of violators spikes first, then time spent poisoned collapses. The group learns to enforce, then individuals learn to comply.
- **The silly rule speeds learning.** Enforcement skill rises faster, and poisoning falls sooner (roughly episodes 3–12).
- **The mechanism switch.** With `transferSkill: false` (skill kept per norm), the silly rule's compliance advantage disappears.
- **Credit assignment.** With `poisonDelay` near 0, agents with no rules learn to avoid pink on their own.

Not robust:

- **The group-welfare advantage.** The silly rule's early punishment spike costs about as many points as faster compliance saves. Some sets of seeds show a significant mid-training gain in group return; others don't replicate it. The paper also found the advantage confined to a middle stretch of training. In this model it depends on the cost of punishing relative to the harm of the poison.

## Tuning notes (what it took to get the effect)

The silly-rule advantage appeared only after three changes, which are worth discussing as claims about when the mechanism should hold:

1. **Enforcement must be the bottleneck.** Skill follows an S-shaped curve in practice, so clumsy enforcers stay clumsy until they've had enough attempts. With easy enforcement, the extra practice buys nothing.
2. **Food must be abundant.** Otherwise putting a second color off-limits is a real food cost. Eating takes several steps (`eatTime`), so travel distance matters little.
3. **Poisoning must last.** With paper-like 1,000-step episodes, a poisoning costs more than a punishment does.

Simplifications relative to the paper: no neural networks or pixels; all 8 agents play every episode (the paper samples 8 of 12); a chase stops once someone else has punished the target.
