// node reports/model_a/webapp/test_engine.js  — checks the JS engine against Python reference outputs
const fs = require("fs"), path = require("path");
const E = require("./sim_engine.js");
const dir = path.join(__dirname, "models");
let fail = 0;
for (const f of fs.readdirSync(dir).filter((x) => x.endsWith(".json") && x !== "index.json")) {
  const J = JSON.parse(fs.readFileSync(path.join(dir, f)));
  const M = new E.Model(J), T = new E.Truth(J), t = J.tests, d = M.d;
  const worst = {};
  const cmp = (name, got, exp) => {
    for (let i = 0; i < exp.length; i++) {
      const e = Math.abs(got[i] - exp[i]) / (Math.abs(exp[i]) + 1e-3);
      worst[name] = Math.max(worst[name] || 0, e);
    }
  };
  for (let n = 0; n < t.z.length; n++) {
    const z = Float64Array.from(t.z[n]), a = t.age[n], u = t.u[n];
    cmp("drift", M.drift(z, a, u, new Float64Array(d)), t.drift[n]);
    cmp("sigma", M.sigma(z, a, new Float64Array(d)), t.sigma[n]);
    cmp("decoded", M.decodeRaw(z, new Float64Array(M.show.length)), t.decoded_raw[n]);
    cmp("log_hazard", [M.logHazard(z)], [t.log_hazard[n]]);
    const zt = Float64Array.from(t.ztrue[n]);
    cmp("true_drift", T.drift(zt, a, u, new Float64Array(8)), t.true_drift[n]);
    cmp("true_sigma", T.sigma(a, new Float64Array(8)), t.true_sigma[n]);
    cmp("true_decoded", T.decodeRaw(zt, new Float64Array(M.show.length)), t.true_decoded_raw[n]);
    cmp("true_log_hazard", [T.logHazard(zt)], [t.true_log_hazard[n]]);
  }
  const bad = Object.entries(worst).filter(([, v]) => v > 1e-3);
  console.log(f, Object.entries(worst).map(([k, v]) => `${k} ${v.toExponential(1)}`).join(", "), bad.length ? "FAIL" : "OK");
  if (bad.length) fail++;
  // timing of one paired scenario run
  const p = J.people[0], K = 48, h = 0.25, n = Math.round((90 - p.age0) / h);
  const eps = E.noise(n, K, d, 1), t0 = Date.now();
  const s0 = E.simulate(M, Float64Array.from(p.z0), p.age0, 90, E.schedule([]), K, h, eps, true);
  const r0 = E.summarise(M, s0, 4);
  console.log(`  simulate+summarise K=${K} ${n} steps: ${Date.now() - t0} ms; RMST to 90 = ${r0.rmst.toFixed(2)} y`);
}
process.exit(fail ? 1 : 0);
