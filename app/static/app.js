async function refreshSignal() {
  const button = document.getElementById("refresh-button");
  button.disabled = true;
  button.textContent = "Refreshing...";
  try {
    const response = await fetch("/api/signal/refresh", { method: "POST" });
    if (!response.ok) {
      throw new Error(`Refresh failed: ${response.status}`);
    }
    const snapshot = await response.json();
    applySnapshot(snapshot);
  } catch (error) {
    console.error(error);
    alert("Unable to refresh the live signal. Check the server logs for details.");
  } finally {
    button.disabled = false;
    button.textContent = "Refresh Signal";
  }
}

function applySnapshot(snapshot) {
  document.getElementById("signal").textContent = snapshot.signal.replaceAll("_", " ").toUpperCase();
  document.getElementById("trade-rationale").textContent = snapshot.trade_rationale;
  document.getElementById("predicted-return").textContent = `${(snapshot.predicted_return * 100).toFixed(3)}%`;
  document.getElementById("absolute-edge-bps").textContent = snapshot.absolute_edge_bps.toFixed(1);
  document.getElementById("confidence").textContent = snapshot.confidence_score.toFixed(1);
  document.getElementById("confidence-label").textContent = snapshot.confidence_label.toUpperCase();
  document.getElementById("forecast-price").textContent = snapshot.forecast_price.toFixed(4);
  document.getElementById("live-last-trade").textContent = snapshot.live_last_trade.toFixed(4);
  document.getElementById("live-bid").textContent = snapshot.live_bid.toFixed(4);
  document.getElementById("live-ask").textContent = snapshot.live_ask.toFixed(4);
  document.getElementById("signal-anchor-price").textContent = snapshot.signal_anchor_price.toFixed(4);
  document.getElementById("xgb").textContent = `${(snapshot.model_breakdown.xgb * 100).toFixed(4)}%`;
  document.getElementById("lgbm").textContent = `${(snapshot.model_breakdown.lgbm * 100).toFixed(4)}%`;
  document.getElementById("ridge").textContent = `${(snapshot.model_breakdown.ridge * 100).toFixed(4)}%`;
  document.getElementById("ensemble").textContent = `${(snapshot.model_breakdown.ensemble * 100).toFixed(4)}%`;
  document.getElementById("threshold").textContent = `${snapshot.threshold_bps.toFixed(1)} bps`;
  document.getElementById("freshness").textContent = `${snapshot.data_freshness_minutes.toFixed(1)} min`;
  document.getElementById("model-version").textContent = snapshot.model_version;
  document.getElementById("brief-provider").textContent = snapshot.market_brief.provider;
  document.getElementById("market-brief").textContent = snapshot.market_brief.content;

  const signalCard = document.querySelector(".signal-card");
  signalCard.className = `signal-card signal-${snapshot.signal}`;

  const features = document.getElementById("top-features");
  features.innerHTML = snapshot.top_features
    .map(
      (feature) =>
        `<li><span>${feature.name}</span><strong>${Number(feature.value).toFixed(5)}</strong></li>`
    )
    .join("");

  const statuses = document.getElementById("source-statuses");
  statuses.innerHTML = snapshot.source_statuses
    .map(
      (status) =>
        `<li class="status-${status.status}"><span>${status.source_id}</span><strong>${status.status.toUpperCase()}</strong><em>${status.message || ""}</em></li>`
    )
    .join("");
}

document.getElementById("refresh-button")?.addEventListener("click", refreshSignal);
