// DriveGuard figures on this site follow the champion model that is scoring the fleet.
// The driveguard repository's daily workflow rewrites app_snapshot/model.json on main whenever the
// champion version changes; this script reads that file and fills every [data-dg] element. The text
// already in the page is the fallback (it is refreshed by the same workflow), so nothing breaks offline.
// Keys mirror scripts/refresh_docs.py in the repository.
(() => {
  const BASE = "https://raw.githubusercontent.com/wenjun78/driveguard/main/app_snapshot/";
  const EPOCH = Date.UTC(1970, 0, 1);
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const dmy = (d) => `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
  const day = (n) => new Date(EPOCH + Number(n) * 86400000);
  const pct = (x, nd) => (x == null ? null : (100 * x).toFixed(nd));
  const num = (x, nd) => (x == null ? null : Number(x).toFixed(nd));
  const int = (x) => (x == null ? null : Math.round(x).toLocaleString("en-US"));

  const split = (p, m, out) => {
    if (!m) return;
    out[`${p}.pr_auc`] = num(m.pr_auc, 3);
    out[`${p}.roc_auc`] = num(m.roc_auc, 3);
    out[`${p}.roc_auc2`] = num(m.roc_auc, 2);
    out[`${p}.drive_recall`] = pct(m.drive_recall_at_budget, 1);
    out[`${p}.drive_recall0`] = pct(m.drive_recall_at_budget, 0);
    out[`${p}.recall`] = pct(m.recall_at_budget, 1);
    out[`${p}.recall0`] = pct(m.recall_at_budget, 0);
    out[`${p}.precision`] = pct(m.precision_at_budget, 1);
    out[`${p}.lead_median`] = int(m.lead_time_median_days);
    out[`${p}.caught_7d`] = pct(m.share_caught_with_7d_lead, 0);
    out[`${p}.false_alerts`] = num(m.false_alerts_per_1000_drive_days, 2);
    out[`${p}.failing_drives`] = int(m.failing_drives);
    out[`${p}.rows_labeled`] = int(m.rows_labeled);
    out[`${p}.rows_labeled_m`] = m.rows_labeled == null ? null : (m.rows_labeled / 1e6).toFixed(2);
    out[`${p}.base_rate`] = m.rows_labeled && m.positives != null ? ((100 * m.positives) / m.rows_labeled).toFixed(2) : null;
    out[`${p}.alerts_per_day`] = int(m.alerts_per_day);
  };

  const values = (model, overview) => {
    const out = { "champion.version": String((model.champion || {}).version ?? "") };
    const r = model.report || {};
    split("val", r.val, out);
    split("test", r.test, out);
    split("rule", r.baseline_test, out);
    const sp = model.split || {};
    if (sp.end_day) {
      const ts = day(sp.test_start), end = day(sp.end_day);
      out["split.train_end"] = dmy(day(sp.train_end));
      out["split.val_start"] = dmy(day(sp.val_start));
      out["split.val_end"] = dmy(day(sp.val_end));
      out["split.test_start"] = dmy(ts);
      out["split.test_end"] = dmy(end);
      out["split.test_days"] = String(Math.round((end - ts) / 86400000) + 1);
      out["split.gap"] = String(sp.gap ?? "");
    }
    const tr = r.test && r.test.recall_at_budget;
    out["live.floor"] = tr ? pct(0.5 * tr, 0) : null;
    if (overview) {
      out["data.drives"] = int(overview.fleet_drives_ever);
      out["data.failures"] = int(overview.fleet_failed_ever);
      out["data.scored_per_day"] = int(overview.drives_scored);
      if (overview.as_of) out["data.as_of"] = dmy(new Date(overview.as_of + "T00:00:00Z"));
      if (overview.generated_at) out["data.exported"] = dmy(new Date(overview.generated_at));
      const mat = (overview.matured || []).map((x) => x.recall_at_budget).filter((x) => x != null).map(Number);
      if (mat.length) {
        const s = mat.slice().sort((a, b) => a - b);
        const med = s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
        out["live.matured_range"] = `${s[0].toFixed(1)}–${s[s.length - 1].toFixed(1)} per day (median ≈ ${med.toFixed(1)})`;
      }
    }
    return out;
  };

  const els = document.querySelectorAll("[data-dg]");
  if (!els.length || !window.fetch) return;
  const get = (f) => fetch(BASE + f, { cache: "no-cache" }).then((x) => (x.ok ? x.json() : null)).catch(() => null);
  Promise.all([get("model.json"), get("overview.json")]).then(([model, overview]) => {
    if (!model) return;
    const v = values(model, overview);
    els.forEach((el) => {
      const val = v[el.dataset.dg];
      if (val != null && val !== "" && el.textContent !== val) el.textContent = val;
    });
    document.querySelectorAll("[data-dg-live]").forEach((el) => (el.hidden = false));
  });
})();
