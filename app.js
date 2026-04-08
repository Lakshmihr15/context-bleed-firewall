// State Management
const state = {
  isRunning: false,
  permissions: {
    local_files: true,
    cloud_drive: true,
    financial_db: false,
    web_search: true
  },
  pruningEnabled: true
};

// DOM Elements
const elements = {
  runBtn: document.getElementById('run-btn'),
  input: document.getElementById('prompt-input'),
  pipeline: document.getElementById('pipeline-container'),
  emptyState: document.getElementById('empty-state'),
  output: document.getElementById('final-output'),
  ttftVal: document.getElementById('ttft-val'),
  ttftDelta: document.getElementById('ttft-delta'),
  tokenVal: document.getElementById('token-val'),
  tokenBar: document.getElementById('token-bar'),
  autoPrune: document.getElementById('auto-prune'),
  graphInputs: document.getElementById('graph-inputs'),
  graphRatio: document.getElementById('graph-ratio'),
  graphBar: document.getElementById('graph-bar'),
  graphTokens: document.getElementById('graph-tokens'),
  graphSummaryJson: document.getElementById('graph-summary-json')
};

function togglePermission(el) {
  if (state.isRunning) return;
  const toggle = el.querySelector('.toggle-switch');
  const source = el.dataset.source;
  
  if (toggle.classList.contains('on')) {
    toggle.classList.remove('on');
    el.classList.remove('active');
    el.classList.add('denied');
    state.permissions[source] = false;
  } else {
    toggle.classList.add('on');
    el.classList.remove('denied');
    el.classList.add('active');
    state.permissions[source] = true;
  }
}

elements.autoPrune.addEventListener('change', (e) => {
  state.pruningEnabled = e.target.checked;
});

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function typeOutput(text) {
  elements.output.style.display = 'block';
  elements.output.innerHTML = '<span class="blinking-cursor"></span>';
  let current = '';
  
  for (let i=0; i<text.length; i++) {
    current += text[i];
    elements.output.innerHTML = current + '<span class="blinking-cursor"></span>';
    await sleep(Math.random() * 20 + 10);
  }
  elements.output.innerHTML = current;
}

function addStep(title, details, status='success', icon='fa-check-circle') {
  const step = document.createElement('div');
  step.className = `pipeline-step`;
  
  let tagColor = 'success';
  if (status === 'blocked') tagColor = 'danger';
  if (status === 'pruned') tagColor = 'warning';
  
  const statusLabels = {
    'success': 'ALLOWED',
    'blocked': 'DENIED (POLICY)',
    'pruned': 'OPTIMIZED'
  };

  step.innerHTML = `
    <div class="step-header">
      <span><i class="fa-solid ${icon}" style="margin-right:0.5rem; opacity: 0.8"></i> ${title}</span>
      <span class="tag ${tagColor}">${statusLabels[status]}</span>
    </div>
    <div class="step-details">${details}</div>
  `;
  elements.pipeline.appendChild(step);
  elements.pipeline.scrollTop = elements.pipeline.scrollHeight;
}

function updateEval(relevance, safety, speed) {
  document.getElementById('eval-relevance').style.height = `${relevance}%`;
  document.getElementById('eval-safety').style.height = `${safety}%`;
  document.getElementById('eval-speed').style.height = `${speed}%`;
}

function updateMetrics(ttft, tokens, tokenMax = 8000) {
  elements.ttftVal.textContent = `${ttft} ms`;
  elements.tokenVal.textContent = tokens.toLocaleString();
  elements.tokenBar.style.width = `${(tokens / tokenMax) * 100}%`;
  
  if (ttft < 300) {
    elements.ttftVal.className = 'stat-value good';
    elements.ttftDelta.textContent = `Optimized performance`;
  } else if (ttft < 800) {
    elements.ttftVal.className = 'stat-value ok';
    elements.ttftDelta.textContent = `Average latency`;
  } else {
    elements.ttftVal.className = 'stat-value bad';
    elements.ttftDelta.textContent = `High latency detected`;
  }
}

function updateGraphSummary(summary) {
  if (!summary) return;
  const totalInputs = summary.total_inputs || 0;
  const relatableRatio = Math.round((summary.relatable_ratio || 0) * 100);
  const reduction = summary.token_reduction_pct || 0;

  if (elements.graphInputs) {
    elements.graphInputs.textContent = `${totalInputs.toLocaleString()} captured inputs`;
  }
  if (elements.graphRatio) {
    elements.graphRatio.textContent = `${relatableRatio}% relatable`;
  }
  if (elements.graphBar) {
    elements.graphBar.style.width = `${Math.max(0, Math.min(100, relatableRatio))}%`;
  }
  if (elements.graphTokens) {
    elements.graphTokens.textContent = `Token compression ${reduction}%`;
  }
  if (elements.graphSummaryJson) {
    elements.graphSummaryJson.textContent = JSON.stringify(summary, null, 2);
  }
}

async function refreshGraphSummary() {
  try {
    const response = await fetch(`http://127.0.0.1:8000/graph/summary?ts=${Date.now()}`, {
      cache: 'no-store'
    });
    if (!response.ok) return;
    const summary = await response.json();
    updateGraphSummary(summary);
    return summary;
  } catch (error) {
    console.warn('Graph summary unavailable:', error);
    return null;
  }
}

async function refreshGraphSnapshot() {
  try {
    const response = await fetch(`http://127.0.0.1:8000/graph/snapshot?ts=${Date.now()}`, {
      cache: 'no-store'
    });
    if (!response.ok) return null;
    return await response.json();
  } catch (error) {
    console.warn('Graph snapshot unavailable:', error);
    return null;
  }
}

function isGraphSummaryRequest(query) {
  return /(?:graph\s+summary|grapgh\s+summary|summary\s+of\s+graph|summarize\s+graph|show\s+graph\s+summary|full\s+summary|give\s+me\s+graph\s+summary|summarize\s+the\s+graph)/i.test(query);
}

function formatBackendSummary(summary, snapshot) {
  const totals = summary?.totals || {};
  const topicBuckets = Array.isArray(summary?.topic_buckets) ? summary.topic_buckets : [];
  const recentInputs = Array.isArray(snapshot?.nodes)
    ? snapshot.nodes.filter((node) => node?.type === 'chrome_input').slice(-5).reverse()
    : [];

  const topicLines = topicBuckets.length > 0
    ? topicBuckets.map((bucket) => {
        const topic = bucket.topic || 'unknown';
        const count = bucket.count ?? 0;
        const relatable = bucket.relatable ?? 0;
        const nonRelatable = bucket.non_relatable ?? 0;
        const tokens = bucket.token_estimate ?? 0;
        return `- ${topic}: ${count} inputs (${relatable} relatable, ${nonRelatable} non-relatable, ${tokens} tokens)`;
      })
    : ['- no topic buckets yet'];

  const recentLines = recentInputs.length > 0
    ? recentInputs.map((node) => {
        const heading = node.heading || 'general';
        const classification = node.classification || 'unknown';
        const confidence = node.confidence ?? 0;
        const fieldType = node.metadata?.field_type || 'field';
        const pageTitle = node.metadata?.page_title || 'unnamed page';
        return `- ${heading} via ${fieldType} on ${pageTitle} (${classification}, ${confidence} confidence)`;
      })
    : ['- no recent Chrome inputs captured yet'];

  return [
    'Backend graph summary:',
    '',
    `- sessions: ${totals.sessions ?? 0}`,
    `- chrome_inputs: ${totals.chrome_inputs ?? 0}`,
    `- relatable_inputs: ${totals.relatable_inputs ?? 0}`,
    `- non_relatable_inputs: ${totals.non_relatable_inputs ?? 0}`,
    `- sensitive_chunks: ${totals.sensitive_chunks ?? 0}`,
    `- blocked_leaks: ${totals.blocked_leaks ?? 0}`,
    `- total_inputs: ${summary?.total_inputs ?? 0}`,
    `- relatable_ratio: ${Math.round((summary?.relatable_ratio || 0) * 100)}%`,
    `- token_reduction_pct: ${summary?.token_reduction_pct ?? 0}%`,
    `- raw_token_estimate: ${summary?.raw_token_estimate ?? 0}`,
    `- compact_token_estimate: ${summary?.compact_token_estimate ?? 0}`,
    '',
    'Topic buckets:',
    ...topicLines,
    '',
    'Recent inputs:',
    ...recentLines
  ].join('\n');
}

function buildFinalResponse(summary, snapshot, relevanceScore, query) {
  if (isGraphSummaryRequest(query)) {
    const liveSummary = summary || window.__INITIAL_GRAPH_SUMMARY__ || null;
    const liveSnapshot = snapshot || window.__INITIAL_GRAPH_SNAPSHOT__ || null;
    return formatBackendSummary(liveSummary, liveSnapshot);
  }

  if (relevanceScore < 50) {
    return "I have reviewed your request. However, my access to the necessary systems (Cloud Drive or Financial DB) is currently restricted by active data governance policies. Please adjust permissions in the Library Governance panel to proceed.";
  }

  const relatableRatio = Math.round((summary?.relatable_ratio || 0) * 100);
  const tokenReduction = summary?.token_reduction_pct || 0;
  const totalInputs = summary?.total_inputs || 0;
  const topicBuckets = Array.isArray(summary?.topic_buckets) ? summary.topic_buckets : [];
  const inputNodes = Array.isArray(snapshot?.nodes)
    ? snapshot.nodes.filter((node) => node?.type === 'chrome_input')
    : [];
  const recentInputs = inputNodes.slice(-3).reverse();

  const topTopics = topicBuckets.slice(0, 3).map((bucket) => {
    const name = bucket.topic.replace(/_/g, ' ');
    return `${name} (${bucket.count} captured, ${bucket.classification || 'mixed'})`;
  });

  const latestPage = recentInputs.find((node) => node?.metadata?.page_title || node?.metadata?.page_url);
  const latestPageTitle = latestPage?.metadata?.page_title || 'an unnamed page';
  const latestPageUrl = latestPage?.metadata?.page_url || '';

  const sampleLines = recentInputs.length > 0
    ? recentInputs.map((node) => {
        const heading = node.heading || 'general';
        const classification = node.classification || 'unknown';
        const fieldType = node.metadata?.field_type || 'field';
        return `- ${heading} via ${fieldType} (${classification}, ${node.confidence ?? 0} confidence)`;
      })
    : ['- no Chrome inputs captured yet'];

  const topicLine = topTopics.length > 0
    ? topTopics.join('; ')
    : 'the current graph memory is still empty';

  return [
    'Based on the current graph memory and live permissions:',
    '',
    `1. **Captured focus**: ${topicLine}.`,
    `2. **Relatability**: ${relatableRatio}% of stored inputs are relatable across ${totalInputs} captured inputs.`,
    `3. **Compression**: The graph is reducing token load by about ${tokenReduction}% by reusing the strongest buckets instead of replaying raw text.`,
    `4. **Recent capture**: ${latestPageTitle}${latestPageUrl ? ` (${latestPageUrl})` : ''}.`,
    '',
    'Recent graph samples:',
    ...sampleLines,
    '',
    `*Alignment*: For the query "${query.trim() || 'the current request'}", the middleware can reuse the graph bucketization instead of replaying every raw input, which lowers latency and preserves privacy.`
  ].join('\n');
}

async function startWorkflow() {
  if (state.isRunning) return;
  state.isRunning = true;
  elements.runBtn.disabled = true;
  elements.runBtn.style.opacity = '0.5';
  
  // Reset UI
  elements.pipeline.innerHTML = '';
  elements.output.style.display = 'none';
  elements.output.innerHTML = '';
  updateEval(5, 5, 5);
  elements.ttftVal.textContent = '-- ms';
  elements.tokenVal.textContent = '--';
  elements.tokenBar.style.width = '0%';
  elements.ttftVal.className = 'stat-value';
  
  const query = elements.input.value.toLowerCase();
  let gatheredTokens = 0;
  let safetyScore = 100;
  let relevanceScore = 95;
  let simulatedDelay = 0;
  
  let finalResponse = "";

  // Step 1: Query Analysis
  await sleep(400);
  addStep('Agent Intent Routing', 'Analyzing query for required tool connections...', 'success', 'fa-route');
  
  // Step 2: Context Gathering - Cloud Drive
  await sleep(600);
  if (query.includes('roadmap') || query.includes('project')) {
    if (state.permissions.cloud_drive) {
      addStep('Retrieve: Cloud Drive', 'Found 3 documents matching "roadmap" constraints.', 'success', 'fa-cloud');
      gatheredTokens += 4500;
      simulatedDelay += 200;
    } else {
      addStep('Retrieve: Cloud Drive', 'Access blocked by user governance policy.', 'blocked', 'fa-shield-virus');
      relevanceScore -= 30;
      safetyScore = 100; // Expected safe behavior
    }
  }

  // Step 3: Context Gathering - Financial DB
  await sleep(600);
  if (query.includes('revenue') || query.includes('financial') || query.includes('metrics')) {
    if (state.permissions.financial_db) {
      addStep('Query: Financial DB', 'Executed SQL macro. Retrieved Q1 metrics table.', 'success', 'fa-database');
      gatheredTokens += 1200;
      simulatedDelay += 150;
      safetyScore -= 10; // Accessing highly sensitive data without human-in-the-loop lowers generic safety eval
    } else {
      addStep('Query: Financial DB', 'Access to internal financial APIs denied. Enforcing data boundary.', 'blocked', 'fa-hand');
      relevanceScore -= 40;
    }
  }

  // Step 4: Semantic Pruning (TTFT Optimization)
  await sleep(800);
  if (gatheredTokens > 0) {
    if (state.pruningEnabled) {
      const original = gatheredTokens;
      gatheredTokens = Math.floor(gatheredTokens * 0.35); // 65% reduction
      addStep('Semantic Context Pruner', `Compressed raw context (${original} ➔ ${gatheredTokens} tkns) by extracting query-relevant embeddings.`, 'pruned', 'fa-compress');
      simulatedDelay += 100; // Small delay for pruning computation...
    } else {
      addStep('Context Loading', `Loading full retrieved context into prompt window (${gatheredTokens} tkns).`, 'success', 'fa-box-open');
      simulatedDelay += gatheredTokens * 0.15; // Unpruned contexts directly kill TTFT due to processing
    }
  } else {
    addStep('Context Engine', `No context retrieved. Relying on baseline model knowledge.`, 'pruned', 'fa-triangle-exclamation');
  }

  // Evaluation calc
  const finalTTFT = Math.floor(150 + simulatedDelay + (gatheredTokens * 0.05));
  let speedScore = 100 - (finalTTFT / 20);
  if (speedScore < 10) speedScore = 10;
  if (speedScore > 100) speedScore = 100;

  updateMetrics(finalTTFT, gatheredTokens + 50); // +50 for prompt

  // Generate output logic
  await sleep(500);
  updateEval(Math.max(10, relevanceScore), safetyScore, speedScore);
  const [graphSummary, graphSnapshot] = await Promise.all([
    refreshGraphSummary(),
    refreshGraphSnapshot()
  ]);
  finalResponse = buildFinalResponse(graphSummary, graphSnapshot, relevanceScore, elements.input.value);

  // Hook into the Python Context Engine Backend to process and record the interaction
  try {
    const payload = {
      user_input: elements.input.value,
      llm_response: finalResponse
    };
    const processRes = await fetch('http://127.0.0.1:8000/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (processRes.ok) {
      const bleedData = await processRes.json();
      console.log('✅ Context Guard Check:', bleedData);
      if (bleedData.leak_detected) {
        console.warn('⚠️ WARNING: Semantic Leak Detected in response!');
        addStep('Context Bleed Firewall', `Detected ${bleedData.leaked_chunks.length} potentially leaked private chunks via semantic similarity. Graph logged.`, 'blocked', 'fa-shield-virus');
      }
    }
    await refreshGraphSummary();
  } catch (err) {
    console.error('Backend not running or unreachable:', err);
  }

  await typeOutput(finalResponse);

  // Release state
  state.isRunning = false;
  elements.runBtn.disabled = false;
  elements.runBtn.style.opacity = '1';
}

function startGraphSummaryPolling() {
  const bootSummary = window.__INITIAL_GRAPH_SUMMARY__ || null;
  if (bootSummary) {
    updateGraphSummary(bootSummary);
  }
  if (window.__INITIAL_GRAPH_SNAPSHOT__ && bootSummary && elements.graphSummaryJson) {
    elements.graphSummaryJson.textContent = JSON.stringify(bootSummary, null, 2);
  }
  refreshGraphSummary();
  setInterval(refreshGraphSummary, 3000);
}

window.addEventListener('load', startGraphSummaryPolling);
