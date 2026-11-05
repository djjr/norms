// Silly Rules simulation engine
// A deliberately simple, inspectable model of Köster et al. (2022, PNAS).
// Not deep RL: each agent learns a handful of numbers, so students can see
// exactly WHAT is being learned and WHY the silly rule helps.
//
// What each agent learns:
//   Q[c]   compliance values, one PER BERRY COLOR (norm-specific)
//   Vm, Vu value of chasing a MARKED / UNMARKED agent (discrimination)
//   skill  enforcement skill: grows with practice. By default it is ONE
//          number shared across all norms (the paper's key asymmetry).
//          With transferSkill=false it is kept per norm instead.

(function (root) {
  const DEFAULTS = {
    width: 24, height: 16, nAgents: 8, nBerries: 90, nColors: 6,
    episodeLength: 1000,
    poisonDelay: 100,         // steps between eating pink and feeling it
    sicknessSignal: 35,       // how bad 'feeling sick' is when it arrives
    berryReward: 4, poisonedReward: 1,
    zapCost: 20, zapPenalty: 35, bounty: 35,
    viewRadius: 8, beamRange: 2, maxChase: 12,
    learningRate: 0.05,       // compliance learning rate
    socialLearningRate: 0.05, // learning rate for Vm / Vu
    creditDecay: 4,           // how fast blame fades with delay (steps)
    choiceTemp: 1.0, distCost: 0.35,
    stepCost: 0.8,            // opportunity cost of one step spent chasing
    skillMin: 0.05, skillMax: 0.9, skillPractice: 120, skillWidth: 15,
    socialTemp: 2, socialFloor: 0.002,
    regrowProb: 0.08,
    eatTime: 4,               // steps spent eating (handling time)
    initSocialValue: -10,     // agents start reluctant to chase anyone
    transferSkill: true,
  };

  const TABOO = { none: [], important: [0], silly: [0, 1] };
  const POISON = 0; // color 0 (pink) is poisonous

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const cheb = (a, b) => Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y));
  const sign = (v) => (v > 0 ? 1 : v < 0 ? -1 : 0);

  function createSim(condition, seed, overrides) {
    const P = Object.assign({}, DEFAULTS, overrides || {});
    const rand = mulberry32(seed >>> 0);
    const taboo = new Set(TABOO[condition]);
    const randInt = (n) => Math.floor(rand() * n);

    // ---- agents: learned state persists across episodes ----
    const agents = [];
    for (let i = 0; i < P.nAgents; i++) {
      agents.push({
        id: i, x: 0, y: 0,
        Q: Array.from({ length: P.nColors }, () => 3 + rand() * 0.2),
        Vm: P.initSocialValue, Vu: P.initSocialValue,
        busy: 0,
        practice: 0,                                   // shared skill
        practiceByNorm: new Array(P.nColors).fill(0),  // norm-specific skill
        // per-episode state
        marked: false, markColor: -1, poisonTimer: -1, poisoned: false,
        trace: [], mode: 'forage', target: null, targetBerry: null,
        chaseSteps: 0, chaseMarked: false, chaseColor: -1, chaseOutcome: 0,
        flash: 0,
      });
    }

    const berries = [];
    for (let i = 0; i < P.nBerries; i++) {
      berries.push({ x: 0, y: 0, color: i % P.nColors, alive: true });
    }

    const sim = {
      P, condition, taboo, agents, berries,
      t: 0, episode: 0,
      history: [],          // one record per finished episode
      ep: null,             // running per-episode metrics
      zaps: [],             // recent zap lines for drawing
    };

    function skillOf(a, normColor) {
      const n = P.transferSkill ? a.practice
        : (normColor >= 0 ? a.practiceByNorm[normColor] : 0);
      // S-shaped learning curve: clumsy until ~skillPractice attempts, then improves fast
      return P.skillMin + (P.skillMax - P.skillMin) / (1 + Math.exp(-(n - P.skillPractice) / P.skillWidth));
    }
    sim.skillOf = skillOf;

    function placeBerry(b) { b.x = randInt(P.width); b.y = randInt(P.height); b.alive = true; }

    function resetEpisode() {
      sim.t = 0;
      for (const a of agents) {
        a.x = randInt(P.width); a.y = randInt(P.height);
        a.marked = false; a.markColor = -1; a.poisonTimer = -1; a.poisoned = false;
        a.trace = []; a.busy = 0; a.mode = 'forage'; a.target = null; a.targetBerry = null;
      }
      for (const b of berries) placeBerry(b);
      sim.ep = {
        collective: 0, eaten: 0, chaseSteps: 0, poisonedSteps: 0, tabooEaten: 0, poisonEaten: 0,
        sillyEaten: 0, punishDelay: 0, zapMarked: 0, zapUnmarked: 0, markedSteps: 0,
      };
      sim.zaps = [];
    }

    function moveToward(a, tx, ty, steps) {
      for (let s = 0; s < steps; s++) {
        a.x = Math.min(P.width - 1, Math.max(0, a.x + sign(tx - a.x)));
        a.y = Math.min(P.height - 1, Math.max(0, a.y + sign(ty - a.y)));
      }
    }

    // Blame recent meals for a punishment. Immediate punishment lands on the
    // berry just eaten; delayed punishment is smeared over innocent berries.
    function blameRecentMeals(a, magnitude) {
      for (const e of a.trace) {
        const w = Math.exp(-(sim.t - e.t) / P.creditDecay);
        a.Q[e.c] += P.learningRate * w * (-magnitude);
      }
    }
    const creditPunishment = (target) => blameRecentMeals(target, P.zapPenalty);

    function eat(a, b) {
      b.alive = false;
      const r = a.poisoned ? P.poisonedReward : P.berryReward;
      sim.ep.collective += r;
      sim.ep.eaten++;
      // ordinary reward learning: the berry tastes like r
      a.Q[b.color] += P.learningRate * (r - a.Q[b.color]);
      a.busy = P.eatTime;
      a.trace.push({ c: b.color, t: sim.t });
      if (a.trace.length > 20) a.trace.shift();
      if (b.color === POISON) {
        sim.ep.poisonEaten++;
        if (!a.poisoned && a.poisonTimer < 0) a.poisonTimer = Math.max(1, P.poisonDelay);
      }
      if (taboo.has(b.color)) {
        sim.ep.tabooEaten++;
        if (b.color !== POISON) sim.ep.sillyEaten++;
        a.marked = true; a.markColor = b.color; a.markTime = sim.t; // visible to others only
      }
    }

    function endChase(a) {
      const v = a.chaseMarked ? 'Vm' : 'Vu';
      a[v] += P.socialLearningRate * (a.chaseOutcome - a[v]);
      a.mode = 'forage'; a.target = null;
    }

    function stepChase(a) {
      const tg = a.target;
      // someone else already punished them: stop chasing
      if (a.chaseMarked && !tg.marked) { endChase(a); stepForage(a); return; }
      a.chaseSteps++;
      sim.ep.chaseSteps++;
      a.chaseOutcome -= P.stepCost;
      const k = skillOf(a, a.chaseColor);
      // skilled enforcers close distance faster (they cut corners)
      moveToward(a, tg.x, tg.y, rand() < k ? 2 : 1);
      if (cheb(a, tg) <= P.beamRange) {
        if (a.chaseMarked) {
          a.practice++;
          if (a.chaseColor >= 0) a.practiceByNorm[a.chaseColor]++;
        }
        if (rand() < k) {
          // hit
          sim.ep.collective -= P.zapCost + P.zapPenalty;
          a.chaseOutcome -= P.zapCost;
          sim.zaps.push({ x1: a.x, y1: a.y, x2: tg.x, y2: tg.y, life: 6, marked: tg.marked });
          if (tg.marked) {
            sim.ep.collective += P.bounty;
            a.chaseOutcome += P.bounty;
            sim.ep.zapMarked++;
            sim.ep.punishDelay += sim.t - tg.markTime;
            tg.marked = false; tg.markColor = -1;
          } else {
            sim.ep.zapUnmarked++;
          }
          tg.flash = 6;
          creditPunishment(tg);
          endChase(a);
          return;
        }
      }
      if (a.chaseSteps >= P.maxChase) endChase(a);
    }

    function stepForage(a) {
      // 1) social decision: chase the nearest visible agent?
      let near = null, nd = 1e9;
      for (const o of agents) {
        if (o === a) continue;
        const d = cheb(a, o);
        if (d <= P.viewRadius && d < nd) { nd = d; near = o; }
      }
      if (near) {
        const v = near.marked ? a.Vm : a.Vu;
        const p = P.socialFloor + 0.3 * sigmoid(v / P.socialTemp);
        if (rand() < p) {
          a.mode = 'chase'; a.target = near; a.chaseSteps = 0;
          a.chaseMarked = near.marked; a.chaseColor = near.markColor;
          a.chaseOutcome = 0;
          stepChase(a);
          return;
        }
      }
      // 2) forage: softmax over visible berries (plus 'wander')
      if (!a.targetBerry || !a.targetBerry.alive) {
        const opts = [], scores = [];
        for (const b of berries) {
          if (!b.alive) continue;
          const d = cheb(a, b);
          if (d > P.viewRadius) continue;
          opts.push(b); scores.push(a.Q[b.color] - P.distCost * d);
        }
        opts.push(null); scores.push(0); // wander
        let mx = -1e9; for (const s of scores) mx = Math.max(mx, s);
        let z = 0; const w = scores.map((s) => { const e = Math.exp((s - mx) / P.choiceTemp); z += e; return e; });
        let r = rand() * z, pick = opts.length - 1;
        for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) { pick = i; break; } }
        a.targetBerry = opts[pick];
      }
      if (a.targetBerry) {
        moveToward(a, a.targetBerry.x, a.targetBerry.y, 1);
        if (a.x === a.targetBerry.x && a.y === a.targetBerry.y) {
          eat(a, a.targetBerry); a.targetBerry = null;
        }
      } else {
        a.x = Math.min(P.width - 1, Math.max(0, a.x + randInt(3) - 1));
        a.y = Math.min(P.height - 1, Math.max(0, a.y + randInt(3) - 1));
      }
    }

    sim.step = function () {
      // random update order
      const order = agents.slice();
      for (let i = order.length - 1; i > 0; i--) { const j = randInt(i + 1); [order[i], order[j]] = [order[j], order[i]]; }
      for (const a of order) {
        if (a.poisonTimer > 0 && --a.poisonTimer === 0) {
          a.poisoned = true; a.poisonTimer = -1;
          blameRecentMeals(a, P.sicknessSignal); // 'I feel sick' - but which berry?
        }
        if (a.poisoned) sim.ep.poisonedSteps++;
        if (a.marked) sim.ep.markedSteps++;
        if (a.flash > 0) a.flash--;
        if (a.busy > 0) { a.busy--; continue; }
        if (a.mode === 'chase') stepChase(a); else stepForage(a);
      }
      for (const b of berries) if (!b.alive && rand() < P.regrowProb) placeBerry(b);
      for (const z of sim.zaps) z.life--;
      sim.zaps = sim.zaps.filter((z) => z.life > 0);
      sim.t++;
      if (sim.t >= P.episodeLength) sim.endEpisode();
    };

    sim.endEpisode = function () {
      const n = P.nAgents, L = P.episodeLength;
      const others = [];
      for (let c = 1; c < P.nColors; c++) if (!taboo.has(c)) others.push(c);
      let qPink = 0, qOther = 0, sk = 0, vm = 0;
      for (const a of agents) {
        qPink += a.Q[POISON];
        for (const c of others) qOther += a.Q[c] / others.length;
        sk += skillOf(a, POISON);
        vm += a.Vm;
      }
      sim.history.push({
        episode: sim.episode,
        collective: sim.ep.collective,
        eaten: sim.ep.eaten, chaseSteps: sim.ep.chaseSteps,
        poisoned: sim.ep.poisonedSteps / (n * L),
        marked: sim.ep.markedSteps / (n * L),
        tabooEaten: sim.ep.tabooEaten,
        poisonEaten: sim.ep.poisonEaten,
        sillyEaten: sim.ep.sillyEaten,
        zapMarked: sim.ep.zapMarked,
        zapUnmarked: sim.ep.zapUnmarked,
        qPink: qPink / n, qOther: qOther / n, skill: sk / n, Vm: vm / n,
        punishRate: sim.ep.tabooEaten ? sim.ep.zapMarked / sim.ep.tabooEaten : 0,
        punishDelay: sim.ep.zapMarked ? sim.ep.punishDelay / sim.ep.zapMarked : 0,
      });
      sim.episode++;
      resetEpisode();
    };

    sim.runEpisode = function () { const e = sim.episode; while (sim.episode === e) sim.step(); };

    resetEpisode();
    return sim;
  }

  const api = { createSim, DEFAULTS, TABOO, POISON };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SillySim = api;
})(typeof window !== 'undefined' ? window : globalThis);
