"use strict";

// Preloaded examples. Scams are clear phishing; the "promo" group are REAL
// held-out India messages verified as W=legit -> A=unwanted (A correct) in
// paper/figures/wa_disagreements.csv. Keeping both shows the two faces of
// "unwanted": outright scam and India promotional spam W lets through.
const EXAMPLES = {
  scam: [
    {
      label: "UPI collect-request",
      text: "ICICI Alert: A UPI collect request of Rs 18,450 is pending. If not initiated by you, verify immediately at icici-secure-review.in",
    },
    {
      label: "Bank KYC",
      text: "Dear customer your bank account will be suspended today. Complete KYC at sbi-kyc-verify.in immediately.",
    },
  ],
  promo: [
    {
      label: "Airtel — FREE/UPI offer",
      text: "FLAT Rs20 CASHBACK for smart savers like you! Available on first 3 UPI transactions only on Airtel Thanks App. Click i.airtel.in/get_Rs20",
    },
    {
      label: "Vi — recharge benefit",
      text: "Rs299 recharged! Enjoy Unlimited Calls+4GB/Day+100SMS/Day. Valid for 28 Days. Your Vi-exclusive Night & Weekend Data benefits activated, click bit.ly/Vi-WDR",
    },
    {
      label: "Vi — pack expiry",
      text: "Alert! Your Vi Unlimited pack EXPIRES in 7 Days. Recharge with best offers: bit.ly/ViRC20 1)299:5GB+4GB/D+UL,28D 2)699:4GB/D+UL,84D",
    },
  ],
  legit: [
    { label: "Personal", text: "Reached home safely. Will call you after dinner." },
    {
      label: "Cab arrival",
      text: "Your cab has arrived. Driver Suresh, vehicle KA01AB1234, is waiting at the gate.",
    },
  ],
};

const $ = (id) => document.getElementById(id);
const SHOWN = 6; // contributions shown before "show all"

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );
}

function verdictClass(v) {
  return v === "unwanted" ? "unwanted" : "legit";
}

function makeChips(items, container, cls) {
  items.forEach((ex) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `chip ${cls}`;
    b.textContent = ex.label;
    b.addEventListener("click", () => {
      $("msg").value = ex.text;
      analyze();
    });
    container.appendChild(b);
  });
}

function renderVerdict(elId, m) {
  const cls = verdictClass(m.verdict);
  const pct = Math.round(m.confidence * 100);
  $(elId).innerHTML = `
    <span class="verdict__badge verdict__badge--${cls}">
      ${cls === "unwanted" ? "⚠" : "✓"} ${m.verdict_label}
    </span>
    <div class="verdict__conf">confidence ${pct}%</div>
    <div class="meter"><div class="meter__fill meter__fill--${cls}" style="width:${pct}%"></div></div>
  `;
}

function whyRow(w, cls, max) {
  const width = Math.max(8, Math.round((w.contribution / max) * 100));
  const kind = w.kind === "term" ? "TERM" : "STRUCT";
  return `
    <li class="why__item">
      <span class="why__term">${escapeHtml(w.feature)}<span class="why__kind why__kind--${w.kind}">${kind}</span></span>
      <span class="why__bar"><span class="why__barfill why__barfill--${cls}" style="width:${width}%"></span></span>
      <span class="why__val">+${w.contribution.toFixed(2)}</span>
    </li>`;
}

function renderTech(elId, m) {
  const cls = verdictClass(m.verdict);
  const pU = Math.round((m.probabilities["unwanted"] || 0) * 100);
  const pL = Math.round((m.probabilities["legitimate"] || 0) * 100);
  let why = "";
  if (!m.why.length) {
    why = `<p class="why__empty">No positive signals — decided by weak/absent cues.</p>`;
  } else {
    const max = Math.max(...m.why.map((w) => w.contribution));
    const head = m.why.slice(0, SHOWN).map((w) => whyRow(w, cls, max)).join("");
    let rest = "";
    if (m.why.length > SHOWN) {
      const more = m.why.slice(SHOWN).map((w) => whyRow(w, cls, max)).join("");
      rest = `<details class="why-more"><summary>Show all ${m.why.length} contributions</summary>
                <ul class="why__list">${more}</ul></details>`;
    }
    why = `<ul class="why__list">${head}</ul>${rest}`;
  }
  $(elId).innerHTML = `
    <div class="probsplit">
      <p class="tech__label">Probability split</p>
      <div class="prob"><span class="prob__name">P(unwanted)</span>
        <span class="prob__bar"><span class="prob__fill prob__fill--unwanted" style="width:${pU}%"></span></span>
        <span class="prob__val">${pU}%</span></div>
      <div class="prob"><span class="prob__name">P(legitimate)</span>
        <span class="prob__bar"><span class="prob__fill prob__fill--legit" style="width:${pL}%"></span></span>
        <span class="prob__val">${pL}%</span></div>
    </div>
    <p class="tech__label">Feature contributions &rarr; ${escapeHtml(m.verdict_label)}</p>
    ${why}
  `;
}

function renderBanner(data) {
  const banner = $("banner");
  const w = data.models.W.verdict;
  let mod = "agree-good";
  if (data.verdicts_differ) mod = "split";
  else if (w === "unwanted") mod = "agree-bad";
  banner.className = `banner banner--${mod}`;
  banner.textContent = data.headline;
}

function renderCallout(data) {
  const callout = $("callout");
  const terms = data.india_terms_learned || [];
  if (!terms.length) {
    callout.hidden = true;
    return;
  }
  $("callout-chips").innerHTML = terms
    .map(
      (t) => `
      <span class="ichip">
        <span class="ichip__term">${escapeHtml(t.feature)}</span>
        <span class="ichip__a">A +${t.a_contribution.toFixed(2)}</span>
        <span class="ichip__w">W ${escapeHtml(t.w_status)}</span>
      </span>`
    )
    .join("");
  callout.hidden = false;
}

function applyLaneColor(laneId, verdict) {
  const lane = $(laneId);
  lane.classList.remove("lane--legit", "lane--unwanted");
  lane.classList.add(`lane--${verdictClass(verdict)}`);
}

async function analyze() {
  const text = $("msg").value.trim();
  if (!text) return;
  const btn = $("analyze");
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    $("echo").textContent = data.message;
    renderBanner(data);
    renderVerdict("verdict-W", data.models.W);
    renderVerdict("verdict-A", data.models.A);
    renderTech("tech-W", data.models.W);
    renderTech("tech-A", data.models.A);
    applyLaneColor("lane-W", data.models.W.verdict);
    applyLaneColor("lane-A", data.models.A.verdict);
    renderCallout(data);
    $("placeholder").hidden = true;
    $("results").hidden = false;
    // Inspection aid: #expand opens every expander (handy for screenshots/viva).
    if (location.hash === "#expand") {
      document.querySelectorAll("details").forEach((d) => (d.open = true));
    }
  } catch (e) {
    $("echo").textContent = "Could not reach the local analysis service.";
    $("placeholder").hidden = true;
    $("results").hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
  }
}

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

async function loadMetrics() {
  try {
    const m = await (await fetch("/metrics")).json();
    if (m.error) return;
    const h = m.headline;
    $("f-india").innerHTML =
      `${h.india_W.toFixed(3)} <span class="arrow">&rarr;</span> ` +
      `<strong>${h.india_A.toFixed(3)}</strong> ` +
      `<span class="delta delta--up">+${h.india_gain.toFixed(3)}</span>`;
    const reg = h.western_delta;
    $("f-west").innerHTML =
      `${h.mendeley_W.toFixed(3)} &rarr; ${h.mendeley_A.toFixed(3)} ` +
      `<span class="delta ${reg < -0.01 ? "delta--down" : "delta--flat"}">` +
      `${reg >= 0 ? "+" : ""}${reg.toFixed(3)}` +
      `${Math.abs(reg) < 0.01 ? " · no regression" : ""}</span>`;
    $("f-novel").innerHTML =
      `<strong>${h.novel_bucket_f1.toFixed(3)}</strong> ` +
      `<span class="muted">(n=${h.novel_bucket_n})</span>`;

    const rows = m.comparison
      .map(
        (r) => `<tr><th scope="row">${r.model}</th>
          <td>${r.mendeley.toFixed(3)}</td><td class="hi">${r.india.toFixed(3)}</td></tr>`
      )
      .join("");
    $("comparison").innerHTML = `
      <table class="cmp">
        <thead><tr><th>Model</th><th>Mendeley test</th><th>India test (held-out)</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <p class="cmp__note">
        5-fold CV macro-F1 — W ${m.cv.W.mean.toFixed(3)}±${m.cv.W.std.toFixed(3)},
        A ${m.cv.A.mean.toFixed(3)}±${m.cv.A.std.toFixed(3)}.
        Train: Mendeley ${m.sizes.mendeley_train} + India ${m.sizes.india_train};
        India test ${m.sizes.india_test} held out (${m.diversity_pct_unique}% unique).
      </p>`;
  } catch (e) {
    /* metrics are optional enrichment; ignore if unavailable */
  }
}

function init() {
  makeChips(EXAMPLES.scam, $("chips-scam"), "chip--scam");
  makeChips(EXAMPLES.promo, $("chips-promo"), "chip--promo");
  makeChips(EXAMPLES.legit, $("chips-legit"), "chip--legit");
  $("analyze").addEventListener("click", analyze);
  $("msg").addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
  });
  loadMetrics();
  // Open on a real disagreement so the finding is visible immediately.
  $("msg").value = EXAMPLES.promo[0].text;
  analyze();
}

document.addEventListener("DOMContentLoaded", init);
