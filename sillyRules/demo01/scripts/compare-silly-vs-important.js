// Compare silly vs important rule on group return, bin by bin, across many seeds.
// Prints the mean difference (silly minus important) and a Welch t-statistic per bin.
// Usage: node scripts/compare-silly-vs-important.js '[{}, {"zapCost":10}]'
//   env EPS=40 SEEDS=16 BASE=2000
const { createSim } = require('../src/engine.js');
const EPS = +(process.env.EPS||40), SEEDS = +(process.env.SEEDS||16), bins = Math.min(10, EPS), per = Math.floor(EPS / bins);
const sets = JSON.parse(process.argv[2] || '[{}]');
for (const over of sets) {
  const curves = {};
  for (const c of ['important', 'silly']) {
    curves[c] = Array.from({ length: bins }, () => []);
    for (let s = 0; s < SEEDS; s++) {
      const sim = createSim(c, (+(process.env.BASE||2000)) + s, over);
      for (let e = 0; e < EPS; e++) sim.runEpisode();
      for (let b = 0; b < bins; b++) {
        let m = 0; for (let e = b * per; e < (b + 1) * per; e++) m += sim.history[e].collective / per;
        curves[c][b].push(m);
      }
    }
  }
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
  const diff = [], tstat = [];
  for (let b = 0; b < bins; b++) {
    const A = curves.silly[b], B = curves.important[b];
    const d = mean(A) - mean(B);
    const se = Math.sqrt(sd(A) ** 2 / A.length + sd(B) ** 2 / B.length);
    diff.push(d.toFixed(0).padStart(6)); tstat.push((d / se).toFixed(1).padStart(6));
  }
  console.log(JSON.stringify(over));
  console.log('  diff', diff.join(''));
  console.log('  t   ', tstat.join(''));
}
