/* Model A in the browser: the exported networks of a trained latent neural SDE, plus the synthetic
 * world's true equation, simulated with Euler–Maruyama. Checked against Python reference outputs by
 * test_engine.js (node). Works as a browser global (MWMEngine) and as a CommonJS module.
 *
 *   dz = f(z, a, u) da + Σ(z, a) dW,   f = −∇V(z) + J(z, a, u)
 */
(function (root) {
  "use strict";

  // ---------------- small dense networks ----------------
  function prepLayers(layers) {
    return layers.map(function (L) {
      var out = L.W.length, inp = L.W[0].length, W = new Float64Array(out * inp);
      for (var i = 0; i < out; i++) for (var j = 0; j < inp; j++) W[i * inp + j] = L.W[i][j];
      return { W: W, b: Float64Array.from(L.b), out: out, inp: inp };
    });
  }
  function affine(L, x, y) {
    for (var i = 0; i < L.out; i++) {
      var s = L.b[i], o = i * L.inp;
      for (var j = 0; j < L.inp; j++) s += L.W[o + j] * x[j];
      y[i] = s;
    }
    return y;
  }
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  function silu(x) { return x * sigmoid(x); }
  function dsilu(x) { var s = sigmoid(x); return s * (1 + x * (1 - s)); }
  function softplus(x) { return x > 20 ? x : Math.log1p(Math.exp(x)); }
  function erf(x) { // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
    var s = x < 0 ? -1 : 1; x = Math.abs(x);
    var t = 1 / (1 + 0.3275911 * x);
    var y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }
  function gelu(x) { return 0.5 * x * (1 + erf(x / Math.SQRT2)); }

  function mlp(layers, act, x, bufs) { // Linear, act, Linear, act, ..., Linear
    var h = x;
    for (var k = 0; k < layers.length; k++) {
      var y = affine(layers[k], h, bufs[k]);
      if (k < layers.length - 1) for (var i = 0; i < y.length; i++) y[i] = act(y[i]);
      h = y;
    }
    return h;
  }

  // ---------------- learned model ----------------
  function Model(json) {
    var w = json.weights;
    this.json = json; this.d = w.d; this.show = w.show;
    this.J = prepLayers(w.J); this.V = prepLayers(w.V); this.D = prepLayers(w.diff);
    this.dec = prepLayers(w.decoder); this.haz = prepLayers(w.hazard); this.hb = w.hazard_bias;
    this.zm = Float64Array.from(w.z_mean); this.zs = Float64Array.from(w.z_std);
    this.sm = w.std_mean; this.ss = w.std_std; this.sl = w.std_log;
    var mk = function (layers) { return layers.map(function (L) { return new Float64Array(L.out); }); };
    this.bJ = mk(this.J); this.bD = mk(this.D); this.bDec = mk(this.dec); this.bHaz = mk(this.haz);
    this.vh1 = new Float64Array(this.V[0].out); this.vh2 = new Float64Array(this.V[1].out);
    this.vg1 = new Float64Array(this.V[0].out); this.vg2 = new Float64Array(this.V[1].out);
    this.inJ = new Float64Array(this.d + 2); this.inD = new Float64Array(this.d + 1);
    this.zr = new Float64Array(this.d);
  }
  Model.prototype.gradV = function (z, g) { // ∇V(z) by hand-written backprop (SiLU MLP 16→128→128→1)
    var V0 = this.V[0], V1 = this.V[1], V2 = this.V[2], h1 = this.vh1, h2 = this.vh2, g1 = this.vg1, g2 = this.vg2, i, j;
    affine(V0, z, h1);
    var s1 = new Float64Array(h1.length); for (i = 0; i < h1.length; i++) s1[i] = silu(h1[i]);
    affine(V1, s1, h2);
    for (i = 0; i < V1.out; i++) g2[i] = V2.W[i] * dsilu(h2[i]);            // dV/dh2 (V2 has one output)
    for (j = 0; j < V1.inp; j++) { var s = 0; for (i = 0; i < V1.out; i++) s += V1.W[i * V1.inp + j] * g2[i]; g1[j] = s * dsilu(h1[j]); }
    for (j = 0; j < V0.inp; j++) { var t = 0; for (i = 0; i < V0.out; i++) t += V0.W[i * V0.inp + j] * g1[i]; g[j] = t; }
    return g;
  };
  Model.prototype.drift = function (z, a, u, out) {
    var d = this.d, x = this.inJ;
    for (var i = 0; i < d; i++) x[i] = z[i];
    x[d] = (a - 60) / 10; x[d + 1] = u;
    var j = mlp(this.J, silu, x, this.bJ);
    var g = this.gradV(z, out);
    for (i = 0; i < d; i++) out[i] = j[i] - g[i];
    return out;
  };
  Model.prototype.sigma = function (z, a, out) {
    var d = this.d, x = this.inD;
    for (var i = 0; i < d; i++) x[i] = z[i];
    x[d] = (a - 60) / 10;
    var s = mlp(this.D, silu, x, this.bD);
    for (i = 0; i < d; i++) out[i] = softplus(s[i]) + 1e-4;
    return out;
  };
  Model.prototype.unwhiten = function (z) { for (var i = 0; i < this.d; i++) this.zr[i] = z[i] * this.zs[i] + this.zm[i]; return this.zr; };
  Model.prototype.decodeRaw = function (z, out) { // typical value (median for log-scale markers)
    var y = mlp(this.dec, gelu, this.unwhiten(z), this.bDec);
    for (var k = 0; k < y.length; k++) { var v = y[k] * this.ss[k] + this.sm[k]; out[k] = this.sl[k] ? Math.exp(v) : v; }
    return out;
  };
  Model.prototype.logHazard = function (z) { return mlp(this.haz, gelu, this.unwhiten(z), this.bHaz)[0] + this.hb; };

  // ---------------- true equation of the synthetic world ----------------
  function Truth(json) {
    var t = json.truth; this.c = t.cfg; this.t = t; this.show = json.weights.show;
    this.on = function (h) { var v = this.c.hallmarks[h]; return v === undefined ? true : !!v; };
  }
  Truth.prototype.inflTarget = function (z0) {
    var c = this.c;
    return this.on("inflammaging") ? c.gamma * c.w * softplus((z0 - c.theta) / c.w) : c.c_lin * (z0 - c.z0_ref);
  };
  Truth.prototype.drift = function (z, a, u, f) {
    var c = this.c, irr = this.on("irreversibility");
    f[0] = irr ? c.beta0 : -c.k_rev * (z[0] - c.c_rev);
    f[1] = -c.k1 * (z[1] - this.inflTarget(z[0]));
    var kap = this.on("nutrient") ? c.kappa : 0;
    f[2] = c.beta2 * (1 - kap * u) - (irr ? 0 : c.k_rev * z[2]);
    f[3] = -c.k_rot * z[3] - c.omega * z[4];
    f[4] = -c.k_rot * z[4] + c.omega * z[3];
    f[5] = this.on("frailty_basin") ? -4 * c.h_frail * z[5] * (z[5] * z[5] - 1) + c.tilt_frail : -c.k5_single * (z[5] + 1);
    f[6] = -c.k_ou[0] * z[6]; f[7] = -c.k_ou[1] * z[7];
    return f;
  };
  Truth.prototype.sigma = function (a, out) {
    var sc = this.on("dispersion") ? 0.4 + 1.2 * Math.min(Math.max((a - 30) / 50, 0), 1.4) : 1;
    for (var i = 0; i < 8; i++) out[i] = sc * this.c.sigma_base[i];
    return out;
  };
  Truth.prototype.sigmaAt = function (z, a, out) { return this.sigma(a, out); };
  Truth.prototype.logHazard = function (z) {
    var c = this.c, lh = c.alpha + c.b0 * z[0] + c.b2 * z[2];
    if (this.on("inflammaging")) lh += c.b1 * z[1];
    if (this.on("frailty_basin")) lh += c.b5 * sigmoid(3 * z[5]);
    return lh;
  };
  Truth.prototype.decodeRaw = function (z, out) { // typical value (median for log-scale markers)
    var t = this.t, zt = new Float64Array(8);
    for (var i = 0; i < 8; i++) zt[i] = (z[i] - t.ref_mean[i]) / t.ref_scale[i];
    for (var k = 0; k < this.show.length; k++) {
      var x = 0, W = t.Wc[k];
      for (i = 0; i < 8; i++) x += zt[i] * W[i];
      x += t.quad[k] * zt[t.prim[k]] * zt[t.prim[k]];
      var cl = t.clin[this.show[k]];
      out[k] = cl.log ? Math.exp(cl.m + cl.s * x) : cl.m + cl.s * x;
    }
    return out;
  };

  // ---------------- random numbers (seeded) ----------------
  function rng(seed) {
    var a = seed >>> 0, spare = null;
    function uni() { a |= 0; a = (a + 0x6D2B79F5) | 0; var t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }
    return function () {
      if (spare !== null) { var s = spare; spare = null; return s; }
      var u, v, r;
      do { u = 2 * uni() - 1; v = 2 * uni() - 1; r = u * u + v * v; } while (r >= 1 || r === 0);
      var m = Math.sqrt(-2 * Math.log(r) / r); spare = v * m; return u * m;
    };
  }
  function noise(n, K, d, seed) { var g = rng(seed), e = new Float64Array(n * K * d); for (var i = 0; i < e.length; i++) e[i] = g(); return e; }

  // u(a) from a list of windows [{start, stop}] (stop may be null = until the end)
  function schedule(windows) {
    return function (a) {
      for (var i = 0; i < windows.length; i++) { var w = windows[i]; if (a >= w.start - 1e-9 && (w.stop === null || w.stop === undefined || a < w.stop - 1e-9)) return 1; }
      return 0;
    };
  }

  // ---------------- simulation ----------------
  // returns {ages, paths: Float64Array((n+1)*K*d), n, K, d}; eps shared between scenarios = paired futures
  function simulate(sys, z0, a0, aEnd, u_of_a, K, h, eps, useNoise) {
    var d = z0.length, n = Math.max(1, Math.round((aEnd - a0) / h)), hh = (aEnd - a0) / n, sq = Math.sqrt(hh);
    var P = new Float64Array((n + 1) * K * d), f = new Float64Array(d), s = new Float64Array(d), z = new Float64Array(d), ages = new Float64Array(n + 1);
    for (var k = 0; k < K; k++) for (var i = 0; i < d; i++) P[k * d + i] = z0[i];
    ages[0] = a0;
    for (var t = 0; t < n; t++) {
      var a = a0 + t * hh, u = u_of_a(a); ages[t + 1] = a0 + (t + 1) * hh;
      for (k = 0; k < K; k++) {
        var o = (t * K + k) * d, o1 = ((t + 1) * K + k) * d;
        for (i = 0; i < d; i++) z[i] = P[o + i];
        sys.drift(z, a, u, f);
        if (sys.sigmaAt) sys.sigmaAt(z, a, s); else sys.sigma(z, a, s);
        for (i = 0; i < d; i++) P[o1 + i] = z[i] + f[i] * hh + (useNoise ? s[i] * sq * eps[o + i] : 0);
      }
    }
    return { ages: ages, paths: P, n: n, K: K, d: d };
  }

  function quantiles(arr, qs) { var s = Array.from(arr).sort(function (a, b) { return a - b; }); return qs.map(function (q) { var x = q * (s.length - 1), i = Math.floor(x), f = x - i; return i + 1 < s.length ? s[i] * (1 - f) + s[i + 1] * f : s[i]; }); }

  // summary of a simulation: marker quantiles at yearly points, survival, mean path
  function summarise(sys, sim, every) {
    var n = sim.n, K = sim.K, d = sim.d, nm = sys.show.length, z = new Float64Array(d), x = new Float64Array(nm);
    var Q = [0.1, 0.25, 0.5, 0.75, 0.9], idx = [], markers = [], i, k, t;
    for (t = 0; t <= n; t += every) idx.push(t);
    if (idx[idx.length - 1] !== n) idx.push(n);
    for (var m = 0; m < nm; m++) markers.push({ q: Q.map(function () { return []; }) });
    var vals = new Float64Array(K * nm);
    idx.forEach(function (tt) {
      for (k = 0; k < K; k++) { var o = (tt * K + k) * d; for (i = 0; i < d; i++) z[i] = sim.paths[o + i]; sys.decodeRaw(z, x); for (m = 0; m < nm; m++) vals[m * K + k] = x[m]; }
      for (m = 0; m < nm; m++) { var qq = quantiles(vals.subarray(m * K, (m + 1) * K), Q); for (var j = 0; j < Q.length; j++) markers[m].q[j].push(qq[j]); }
    });
    // survival: S(a) = mean_k exp(−∫ λ), λ = exp(log hazard) per year, trapezoid rule
    var hh = sim.ages[1] - sim.ages[0], cum = new Float64Array(K), lamPrev = new Float64Array(K), S = new Float64Array(n + 1), lam90 = [];
    for (k = 0; k < K; k++) { var o0 = k * d; for (i = 0; i < d; i++) z[i] = sim.paths[o0 + i]; lamPrev[k] = Math.exp(sys.logHazard(z)); }
    S[0] = 1;
    for (t = 1; t <= n; t++) {
      var sm = 0;
      for (k = 0; k < K; k++) { var o2 = (t * K + k) * d; for (i = 0; i < d; i++) z[i] = sim.paths[o2 + i]; var l = Math.exp(sys.logHazard(z)); cum[k] += 0.5 * (l + lamPrev[k]) * hh; lamPrev[k] = l; sm += Math.exp(-cum[k]); }
      S[t] = sm / K;
    }
    var rmst = 0; for (t = 1; t <= n; t++) rmst += 0.5 * (S[t] + S[t - 1]) * hh;
    var mean = new Float64Array((n + 1) * d);
    for (t = 0; t <= n; t++) for (k = 0; k < K; k++) { var o3 = (t * K + k) * d; for (i = 0; i < d; i++) mean[t * d + i] += sim.paths[o3 + i] / K; }
    return { ages: Array.from(sim.ages), markerAges: idx.map(function (tt) { return sim.ages[tt]; }), markers: markers, survival: Array.from(S), rmst: rmst, mean: mean };
  }

  // projection of latent points onto a landscape plane, lifted onto its surface
  function lift(plane, z, out) {
    var x = 0, y = 0, n = z.length;
    for (var i = 0; i < n; i++) { var c = z[i] - plane.origin[i]; x += c * plane.w[i]; y += c * plane.e2[i]; }
    var X = plane.x, Y = plane.y, nx = X.length, ny = Y.length;
    var fx = (Math.min(Math.max(x, X[0]), X[nx - 1]) - X[0]) / (X[1] - X[0]), fy = (Math.min(Math.max(y, Y[0]), Y[ny - 1]) - Y[0]) / (Y[1] - Y[0]);
    var i0 = Math.min(nx - 2, Math.floor(fx)), j0 = Math.min(ny - 2, Math.floor(fy)), tx = fx - i0, ty = fy - j0, U = plane.U;
    var h = U[i0][j0] * (1 - tx) * (1 - ty) + U[i0 + 1][j0] * tx * (1 - ty) + U[i0][j0 + 1] * (1 - tx) * ty + U[i0 + 1][j0 + 1] * tx * ty;
    out[0] = x; out[1] = y; out[2] = h;
    return out;
  }

  var api = { Model: Model, Truth: Truth, rng: rng, noise: noise, schedule: schedule, simulate: simulate, summarise: summarise, lift: lift, quantiles: quantiles };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.MWMEngine = api;
})(typeof self !== "undefined" ? self : this);
