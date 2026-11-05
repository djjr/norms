// Run all three conditions headless and print binned averages of each metric.
// Usage: node scripts/run-conditions.js [episodes=200] [populations=8] ['{"param":value}']
//   env KEYS=collective,poisoned,...   choose which metrics to print
//   env BASE=1000                      first random seed
const { createSim } = require('../src/engine.js');
const EPS = +process.argv[2] || 200, SEEDS = +process.argv[3] || 8;
const over = process.argv[4] ? JSON.parse(process.argv[4]) : {};
const conds = ['none', 'important', 'silly'];
const res = {};
const t0 = Date.now();
for (const c of conds) {
  res[c] = [];
  for (let s = 0; s < SEEDS; s++) {
    const sim = createSim(c, (+(process.env.BASE||1000)) + s, over);
    for (let e = 0; e < EPS; e++) sim.runEpisode();
    res[c].push(sim.history);
  }
}
const bins = Math.min(10, EPS), per = Math.floor(EPS / bins);
const keys = process.env.KEYS ? process.env.KEYS.split(',') : ['collective','poisoned','zapMarked','skill','Vm','punishRate','punishDelay','qPink'];
for (const k of keys) {
  console.log('\n' + k);
  for (const c of conds) {
    const row = [];
    for (let b = 0; b < bins; b++) {
      let sum = 0, n = 0;
      for (const h of res[c]) for (let e = b * per; e < (b + 1) * per; e++) { sum += h[e][k]; n++; }
      row.push((sum / n).toFixed(['poisoned','skill','punishRate'].includes(k) ? 2 : 1).padStart(8));
    }
    console.log(c.padEnd(10), row.join(''));
  }
}
console.log('\ntime', (Date.now() - t0) / 1000, 's');
