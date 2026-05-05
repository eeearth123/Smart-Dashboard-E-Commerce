<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Churn Intelligence Platform — Business Impact</title>
  <style>
    :root {
      --bg: #ffffff;
      --bg-alt: #f6f8fa;
      --text: #1f2328;
      --text-light: #656d76;
      --primary: #0969da;
      --success: #1a7f37;
      --warning: #9a6700;
      --danger: #cf222e;
      --border: #d0d7de;
      --radius: 8px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
      line-height: 1.6;
      color: var(--text);
      background: var(--bg);
      max-width: 860px;
      margin: 0 auto;
      padding: 2rem 1.5rem;
    }
    a { color: var(--primary); text-decoration: none; }
    a:hover { text-decoration: underline; }
    
    header {
      border-bottom: 1px solid var(--border);
      padding-bottom: 1.5rem;
      margin-bottom: 2rem;
    }
    h1 {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1.2;
      margin-bottom: 0.5rem;
    }
    .highlight { color: var(--success); font-weight: 700; }
    .subtitle { font-size: 1.15rem; color: var(--text-light); font-weight: 400; }
    
    section { margin-bottom: 2rem; }
    h2 {
      font-size: 1.4rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    p, li { font-size: 1rem; margin-bottom: 0.5rem; }
    ul { margin-left: 1.5rem; margin-bottom: 1rem; }
    li { margin-bottom: 0.4rem; }
    
    .approach-steps {
      background: var(--bg-alt);
      border-left: 4px solid var(--primary);
      padding: 1rem 1.2rem;
      border-radius: 0 var(--radius) var(--radius) 0;
      margin: 1rem 0;
    }
    .approach-steps ol { margin-left: 1.2rem; }
    .approach-steps li { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.95rem; }
    
    .results-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
      margin: 1rem 0;
    }
    @media (max-width: 640px) { .results-grid { grid-template-columns: 1fr; } }
    
    .result-card {
      background: var(--bg-alt);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.2rem;
    }
    .result-card.untargeted { border-left: 4px solid var(--warning); }
    .result-card.targeted { border-left: 4px solid var(--success); }
    .result-card h3 { font-size: 1.1rem; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem; }
    .metric { font-weight: 600; color: var(--text); }
    .metric.roi-low { color: var(--warning); }
    .metric.roi-high { color: var(--success); }
    
    .callout {
      padding: 1rem 1.2rem;
      border-radius: var(--radius);
      margin: 1rem 0;
      font-size: 0.95rem;
    }
    .callout.info { background: #ddf4ff; border-left: 4px solid var(--primary); }
    .callout.warn { background: #fff8c5; border-left: 4px solid var(--warning); }
    .callout.success { background: #dafbe1; border-left: 4px solid var(--success); }
    
    .docs-nav {
      background: var(--bg-alt);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem 1.2rem;
      margin-top: 2rem;
    }
    .docs-nav h3 { font-size: 1rem; margin-bottom: 0.5rem; }
    .docs-nav a { display: block; padding: 0.4rem 0; font-weight: 500; }
    .docs-nav a::before { content: "📄 "; }
    
    footer {
      margin-top: 3rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      text-align: center;
      font-size: 0.85rem;
      color: var(--text-light);
    }
    
    /* Print Friendly */
    @media print {
      body { padding: 0; max-width: none; }
      a { text-decoration: none; color: #000; }
      .result-card { break-inside: avoid; }
    }
  </style>
</head>
<body>

  <header>
    <h1>💰 Same campaign. Same cost per customer.<br><span class="highlight">78x better ROI.</span></h1>
    <p class="subtitle">But here's the key: we didn't just predict churn — we simulated what would happen if we intervened.</p>
  </header>

  <main>
    <section id="problem">
      <h2>📉 The Problem</h2>
      <p>We built a churn prediction model (LightGBM, 58K orders).<br>
      Initial idea: offer free shipping to everyone.<br>
      <strong>But would it work? And for whom?</strong></p>
    </section>

    <section id="approach">
      <h2>🧠 The Approach</h2>
      <div class="approach-steps">
        <ol>
          <li>Take customer's feature vector</li>
          <li>Modify feature (<code>set freight_value = 0</code>)</li>
          <li>Re-run <code>predict_proba()</code></li>
          <li>Calculate ROI: <code>[(Value of saved customers - Budget) / Budget] × 100</code></li>
        </ol>
      </div>
    </section>

    <section id="segmentation">
      <h2>🎯 Strategic Segmentation</h2>
      <p>Instead of treating all predictions equally, we combined:</p>
      <ul>
        <li>AI confidence (probability thresholds)</li>
        <li>Business context (category profitability)</li>
        <li>Customer behavior patterns</li>
      </ul>
      <p><strong>Different segments → Different strategies</strong></p>
    </section>

    <section id="results">
      <h2>🔢 Results</h2>
      <div class="results-grid">
        <div class="result-card untargeted">
          <h3>🟡 Untargeted (45,791 customers)</h3>
          <ul>
            <li><strong>Budget:</strong> 45,791 × R$34 = ~R$895,118 (total campaign cost)</li>
            <li><strong>Success Rate:</strong> <span class="metric">24.7%</span> (6,491 customers flipped from Churn → Stay)</li>
            <li><strong>Break-even:</strong> <span class="metric">23.6%</span> (R$34 cost / R$144 avg. order value)</li>
            <li><strong>ROI:</strong> <span class="metric roi-low">+3.4%</span> (barely above break-even)</li>
            <li><strong>Result:</strong> Profitable, but with very low gross profit and high operating costs.</li>
          </ul>
        </div>
        
        <div class="result-card targeted">
          <h3>🟢 ML-Targeted (5,363 high-risk + profitable categories)</h3>
          <ul>
            <li><strong>Budget:</strong> 5,363 × R$34 = R$25,112 (97% less!)</li>
            <li><strong>Success Rate:</strong> <span class="metric">46.4%</span> (model + targeting = higher response)</li>
            <li><strong>Break-even:</strong> <span class="metric">13.5%</span> (higher avg. order value segment = easier to profit)</li>
            <li><strong>ROI:</strong> <span class="metric roi-high">+266.6%</span> (much higher margin above break-even)</li>
            <li><strong>Result:</strong> Same campaign, dramatically better outcome.</li>
          </ul>
        </div>
      </div>
    </section>

    <section id="takeaway">
      <h2>💡 The Takeaway</h2>
      <div class="callout info">
        The model doesn't just predict <em>who will churn</em> — it identifies <em>who will actually respond</em> to a specific intervention.
      </div>
    </section>

    <section id="caveat">
      <h2>⚠️ Caveat</h2>
      <div class="callout warn">
        Numbers depend on model quality (ours: <strong>Macro F1 = 0.658</strong>), feature engineering, and simulation assumptions. Always validate with real-world A/B tests before full rollout.
      </div>
    </section>

    <section id="final-thoughts">
      <h2>📊 Final Thoughts</h2>
      <ul>
        <li><strong>Accuracy:</strong> 74.74%</li>
        <li><strong>Macro F1:</strong> 0.6742</li>
      </ul>
      <div class="callout success">
        Metrics alone don't drive action. If we can't explain <strong>WHY</strong>, business teams won't trust it.<br><br>
        That's why we used:<br>
        ✅ SHAP values<br>
        ✅ Feature importance analysis<br>
        ✅ Probability distribution visualization<br><br>
        <em>A model is only as good as your ability to explain it.</em>
      </div>
    </section>

    <div class="docs-nav">
      <h3>📚 Deep Dive Documentation</h3>
      <a href="Docs/MODEL.md">🔍 Model Architecture, Metrics & SHAP Interpretation</a>
      <a href="Docs/DASHBOARD.md">📈 Dashboard Walkthrough, UI Flow & Business Insights</a>
    </div>
  </main>

  <footer>
    <p>Built with Python · LightGBM · SHAP · Streamlit · BigQuery</p>
    <p>© 2025–2026 Thanyathorn Krutphan | Available under MIT License</p>
  </footer>

</body>
</html>
