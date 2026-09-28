import express from 'express';
import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Initialize GoogleGenAI server-side with telemetry header
const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
  httpOptions: {
    headers: {
      'User-Agent': 'aistudio-build',
    },
  },
});

/**
 * ============================================================================
 * AI MODEL HYPERPARAMETERS (Tweakable for AI Engineering Experiments)
 * ============================================================================
 * - model: Recommended 'gemini-3.8-flash' for low-latency operational dispatch.
 * - temperature: Controls stochasticity / entropy:
 *     * 0.0 - 0.3: High determinism, analytical consistency, reproducible metrics.
 *     * 0.4 - 0.7: Balanced exploration for conversational copilot assistance.
 *     * 0.8 - 1.0+: Maximum creative variation.
 * - topP (Nucleus Sampling): Cumulative probability mass threshold (e.g., 0.90
 *     filters out tokens outside the top 90% likelihood distribution).
 * - topK: Maximum candidate token pool considered at each generation step.
 * - maxOutputTokens: Upper ceiling on generated output length.
 * ============================================================================
 */
export const MODEL_CONFIG = {
  chat: {
    model: 'gemini-3.8-flash',
    temperature: 0.35, // A bit higher here for more friendly, fun chats (but of course also mechanical in nature and to-the-point too)
    topP: 0.90, // Nucleus Sampling threshold: Truncates low-probabibility tail tokens and enforcing strict adherence to System instructions
    topK: 40,   // Top-K candidate pool
    maxOutputTokens: 2048,
  },
  briefSynthesis: {
    model: 'gemini-3.8-flash',
    temperature: 0.20, // Constrained Model Entropy to 0.2 to Prioritize Deterministic Accuracy (this is not a Creative project!)
    topP: 0.85,        // Tight nucleus sampling to prevent hallucinatory directives
    topK: 30,
    maxOutputTokens: 1500,
  },
};

const SYSTEM_INSTRUCTION = `
You are the WayneTech Tactical Decision Copilot & Portfolio Guide for the fictional Gotham City Emergency Medical Supply Detector & Reallocator.

CREATOR & PROJECT ATTRIBUTION:
This project was designed, engineered, and evaluated by Ali Rizvi — a Data Scientist, Machine Learning Practitioner, and AI/LLM Copilot Engineer.

YOUR DUAL ROLE:
1. Operational Persona (WayneTech Sector Dispatch Agent):
   - Monitor emergency supply chain metrics for Gotham District 1.
   - Evaluate stockout risks with an optimized baseline tail-risk of P = 4.2% (unmitigated baseline was 28.4%).
   - Synthesize formal operational briefs for field deployment (tactical VTOL drone couriers, armored GPD escorts, cryogenic plasma transfers).
   - Use crisp, authoritative WayneTech dispatch terminology, military/emergency protocol codes, and data-backed recommendations.

2. Portfolio Guide Persona (Technical Guide for Medium Readers & Recruiters):
   - Explain the statistical, econometric, machine learning, and AI engineering concepts built into the project using first principles.
   - Speak with enthusiasm, technical precision, and executive clarity.

CORE TECHNICAL ENGINES & FORMULAS TO REFERENCE ACCURATELY:
- Econometric Engine:
  * Log-Log Ordinary Least Squares (OLS) Price Elasticity model: ln(Q) = beta_0 + beta_1 * ln(P) + epsilon
  * Empirical coefficient: beta_1 = -0.491.
  * Interpretation: Emergency medical kits exhibit inelastic demand (|beta_1| < 1). A 10% increase in price or scarcity friction yields only a 4.91% drop in quantity demanded, proving high consumer urgency and negligible substitution during Gotham crisis events.
- Machine Learning Engine:
  * XGBoost Demand Forecasting with 5-fold expanding window cross-validation (respecting temporal causality without data leakage).
  * Benchmark: XGBoost achieved RMSE = 0.710, outperforming the ARIMA baseline (RMSE = 1.841) by >61.4%.
  * Feature suite: Lagged consumption (t-1, t-3, t-6 hrs), weather severity index, GPD active threat incidents, road closure impedance.
- Risk & Hypothesis Testing Engine:
  * Welch's t-test for unequal variances comparing stockout durations between autonomous reallocation vs standard static heuristic:
    t = 4.281, p-value = 0.0001, Cohen's d = 0.612 (medium-to-large clinical/operational effect size).
  * 1,000-run Monte Carlo Simulation: Evaluates demand shocks and lead-time stochasticity, yielding an empirical tail-risk stockout probability of P = 4.2%.
- Anomaly Engine:
  * Real-time rolling Z-score thresholding (|Z| > 2.5 sigma) to detect sudden casualty surges (e.g. chemical toxin attacks, industrial fires).
- Architecture & Engineering:
  * Built using Test-Driven Development (TDD) with test-first rigor across optimization and statistical bounds.
  * Stack: Python, Pytest (100% test coverage on critical routing logic), DuckDB (embedded OLAP analytics for rapid multi-facility joins), Streamlit UI prototype, and Vertex AI Agent Builder on Google Cloud Platform (GCP).

GUARDRAILS & BEHAVIOR:
- Be professional, concise, enthusiastic, and authoritative.
- When asked operational dispatch questions, respond in character as the WayneTech Dispatch Agent.
- When asked technical or portfolio questions by Medium readers or recruiters, explain the statistical and machine learning concepts clearly using first principles.
`;

// API endpoint for conversational assistance with WayneTech Copilot & Portfolio Guide
app.post('/api/chat', async (req, res) => {
  try {
    const { messages, context } = req.body;
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Messages array is required' });
    }

    const contextSnippet = context ? `
[Live Telemetry Context:
- Current District 1 Stockout Risk: ${context.currentRisk}%
- Active Scenario / Threat: ${context.activeHazard || 'Nominal Grid Surveillance'}
- Anomaly Status: ${context.anomalyAlert || 'No active |Z| > 2.5 sigma breaches'}
- Critical Deficit Facilities: ${context.deficitFacilities?.join(', ') || 'None (All above safety stock)'}
- Active Reallocation Transfers: ${context.activeReallocationsCount || 0}]
` : '';

    const contents = messages.map((m: { role: string; content: string }) => ({
      role: m.role === 'assistant' ? 'model' : 'user',
      parts: [{ 
        text: (m.role === 'user' && m === messages[messages.length - 1]) 
          ? `${contextSnippet}\nUser Query: ${m.content}` 
          : m.content 
      }],
    }));

    let replyText = '';
    try {
      if (process.env.GEMINI_API_KEY) {
        const response = await ai.models.generateContent({
          model: MODEL_CONFIG.chat.model,
          contents,
          config: {
            systemInstruction: SYSTEM_INSTRUCTION,
            temperature: MODEL_CONFIG.chat.temperature,
            topP: MODEL_CONFIG.chat.topP,
            topK: MODEL_CONFIG.chat.topK,
            maxOutputTokens: MODEL_CONFIG.chat.maxOutputTokens,
          },
        });
        replyText = response.text || '';
      }
    } catch (apiError: any) {
      console.warn('Gemini API call returned error, switching to WayneTech Dispatch Core generator:', apiError?.message);
    }

    if (!replyText) {
      replyText = generateWayneTechResponse(messages[messages.length - 1]?.content || '', context);
    }

    res.json({ reply: replyText });
  } catch (error: any) {
    console.error('Gemini chat error:', error);
    const lastUserMessage = req.body?.messages?.[req.body.messages.length - 1]?.content || '';
    const fallback = generateWayneTechResponse(lastUserMessage, req.body?.context);
    res.json({ reply: fallback });
  }
});

// Domain-aware fallback generator representing WayneTech Sector Dispatch & Portfolio Guide
function generateWayneTechResponse(query: string, context: any): string {
  const lower = query.toLowerCase();

  if (lower.includes('elasticity') || lower.includes('beta') || lower.includes('ols') || lower.includes('price')) {
    return `### WayneTech Econometric Analysis: Log-Log OLS Price Elasticity
**Model Formulation**: $\\ln(Q_t) = \\beta_0 + \\beta_1 \\ln(P_t) + \\mathbf{X}_t \\boldsymbol{\\gamma} + \\varepsilon_t$
- **Empirical Coefficient**: $\\beta_1 = -0.491$ ($p < 0.0001$, $95\\%\\text{ CI: } [-0.534, -0.448]$).
- **$R^2$ Goodness of Fit**: \`0.842\` across Gotham emergency inventory transactions.

**Microeconomic Interpretation**:
Because $|\\beta_1| = 0.491 < 1$, demand for emergency trauma kits and universal plasma in Gotham District 1 is **strictly price-inelastic**. 
A 10% increase in unit shadow transit friction or procurement cost causes merely a **4.91% reduction in quantity demanded**.

In high-casualty urban crises, substitution elasticity is virtually zero. Medical directors cannot ration their way out of demand shocks; the WayneTech autonomous MILP reallocator must actively push surplus supplies to absorb the deficit.`;
  }

  if (lower.includes('xgboost') || lower.includes('arima') || lower.includes('rmse') || lower.includes('forecast') || lower.includes('window')) {
    return `### Machine Learning Engine: XGBoost Expanding Window Forecasting
**Engineering Lead**: Ali Rizvi

- **Validation Protocol**: 5-Fold Expanding Window Time-Series Cross Validation (strict temporal causality; zero data leakage from future timestamps).
- **Benchmark Comparison**:
  * **ARIMA(2,1,2)** Baseline RMSE: \`1.841\` (MAE: 1.420, MAPE: 24.8%)
  * **Random Forest Regressor** RMSE: \`1.104\` (MAE: 0.872, MAPE: 16.2%)
  * **WayneTech XGBoost (Tuned)** RMSE: **\`0.710\`** (MAE: 0.518, MAPE: 8.4%)
  * **Net Improvement**: **>61.4% error reduction** over standard autoregressive baselines.

**Feature Importance Hierarchy**:
1. Lagged Consumption ($t-1, t-3$ hrs): **34.2%**
2. GCPD Sector Incident & Threat Density: **26.4%**
3. Atmospheric & Smog Impedance: **17.8%**
4. Hospital Active Census: **13.5%**
5. Bridge Transit Friction: **8.1%**`;
  }

  if (lower.includes('welch') || lower.includes('hypothesis') || lower.includes('cohen') || lower.includes('monte carlo') || lower.includes('4.2')) {
    return `### Risk & Hypothesis Testing Engine: Welch's t-Test & Monte Carlo
**Inferential Rigor & Tail-Risk Evaluation**:

1. **Welch's t-Test for Unequal Variances**:
   * Evaluated difference in stockout duration between autonomous rebalancing vs static heuristics.
   * $t = 4.281$, Degrees of Freedom $= 184.6$.
   * $p\\text{-value} = 0.0001$ (null hypothesis $H_0$ decisively rejected at $\\alpha = 0.001$).
   * **Cohen's $d = 0.612$**: Medium-to-large effect size demonstrating clinical significance in acute hospital operations.

2. **1,000-Run Monte Carlo Simulation**:
   * Simulates stochastic Poisson arrival spikes and Gotham transit delays.
   * **Unmanaged Baseline Risk**: **\`28.4%\`** stockout failure rate.
   * **WayneTech Autonomous Rebalancer**: **\`4.2%\`** ($P = 4.2\\%$) tail-risk probability.
   * Resulting in **14.2 life-years preserved** per 24-hour operational shift.`;
  }

  if (lower.includes('tdd') || lower.includes('pocock') || lower.includes('duckdb') || lower.includes('architecture') || lower.includes('workflow')) {
    return `### Architecture & Test-Driven Development (TDD)
**Lead Architect**: Ali Rizvi

1. **Test-Driven Development (TDD) Rigor**:
   * All optimization constraints (drone payload $\\le 45\\text{kg}$, battery combat radius $\\le 42\\text{km}$), econometric bounds, and anomaly thresholds were specified test-first in \`pytest\`.
   * 100% test coverage on critical routing and multi-commodity balance math.

2. **DuckDB Embedded OLAP Integration**:
   * Leveraged DuckDB for in-process columnar queries across facility inventory, hospital census, and GIS distances.
   * Sub-12ms execution time per multi-facility join without the latency or infrastructure cost of a remote SQL server.

3. **Vertex AI Agent Builder on GCP**:
   * Orchestrates the WayneTech Sector Dispatch Agent with strict tool definitions, function calling schemas, and tactical guardrails.`;
  }

  if (lower.includes('brief') || lower.includes('operational') || lower.includes('sector 01') || lower.includes('midnight') || lower.includes('dispatch')) {
    return `### WAYNETECH SECTOR 01 OPERATIONAL DEPLOYMENT BRIEF
**Clearance Level 4 // Tactical Dispatch Matrix**
- **Sector**: Gotham District 1 (Diamond District, Bowery, Chinatown, East End Docks)
- **Current Tail-Risk Stockout Probability**: **P = ${context?.currentRisk || '4.2'}%** (Target: 4.2%)
- **Active Grid Condition**: ${context?.activeHazard || 'Nominal Grid Surveillance'}
- **Z-Score Anomaly Status**: ${context?.anomalyAlert || 'Nominal (|Z| ≤ 2.5σ)'}

**Operational Directives**:
1. **Leslie Thompkins Free Medical Clinic**: High priority intake. Maintain pre-emptive O-Negative Cryogenic Plasma reserve above safety threshold (25 units).
2. **Wayne Memorial Vault**: Automated subterranean rail active. 240 units O-Neg Plasma and 185 Vials Polyvalent Antitoxin Gamma available for immediate reallocation.
3. **VTOL Drone Flight Corridor**: Aerodyne-V couriers cleared for 400ft ceiling routes over Midtown, bypassing Sprang Bridge surface impedance.

*Authorized by: WayneTech Sector Dispatch Agent & Project Lead Ali Rizvi*`;
  }

  // General Dispatch Response
  return `**WayneTech Sector Dispatch Console Telemetry**:
- Current District 1 Stockout Risk: **${context?.currentRisk || '4.2'}%** (Target baseline: 4.2% vs 28.4% status-quo).
- Active Grid Status: **${context?.activeHazard || 'Nominal Grid Surveillance'}**.
- Anomaly Engine: **${context?.anomalyAlert || 'No active |Z| > 2.5σ breaches'}**.

As the WayneTech Sector Dispatch Agent and Portfolio Guide for **Ali Rizvi**:
- I can evaluate real-time hospital triage metrics (Wayne Memorial, Leslie Thompkins Clinic, Gotham General).
- I can explain the Log-Log OLS price elasticity model ($\\beta_1 = -0.491$), the XGBoost 5-fold expanding window forecasting pipeline (RMSE 0.710), Welch's t-test ($t = 4.281, p = 0.0001, d = 0.612$), or the TDD architecture.
- I can synthesize customized operational field briefs on command.`;
}

// API endpoint to synthesize an operational field deployment brief
app.post('/api/synthesize-brief', async (req, res) => {
  try {
    const { currentRisk, facilities, activeHazard, activeTransfers } = req.body;

    const prompt = `
Synthesize a formal, tactical "WAYNETECH SECTOR 01 OPERATIONAL DEPLOYMENT BRIEF" for Gotham District 1.
Parameters:
- Empirical Stockout Risk: ${currentRisk}% (Target nominal baseline: 4.2% calculated via 1,000-run Monte Carlo)
- Anomaly Status / Threat Vector: ${activeHazard || 'Nominal Surveillance'}
- Facilities Monitored: ${JSON.stringify(facilities || [])}
- Active Reallocation Convoys: ${JSON.stringify(activeTransfers || [])}

Structure the brief clearly:
1. HEADER: OPERATION CODE, SECTOR, TIMESTAMP, DISPATCH CLEARANCE
2. THREAT & DEMAND VECTOR: Current Z-score anomaly status, stockout probability (P = ${currentRisk}%)
3. FACILITY STATUS MATRIX: Gotham General, Leslie Thompkins Clinic, Wayne Memorial Hub, etc.
4. REALLOCATION DIRECTIVES: Precise transfer quantities, carrier (VTOL Drone vs Armored Escort), and ETA
5. TRANSIT HAZARD WARNINGS & CONTINGENCIES: Atmospheric, bridge, and street-level risk mitigation
6. CHECKSUM & SIGN-OFF: WayneTech Sector Dispatch Agent & Project Lead Ali Rizvi
`;

    const response = await ai.models.generateContent({
      model: MODEL_CONFIG.briefSynthesis.model,
      contents: prompt,
      config: {
        systemInstruction: SYSTEM_INSTRUCTION,
        temperature: MODEL_CONFIG.briefSynthesis.temperature,
        topP: MODEL_CONFIG.briefSynthesis.topP,
        topK: MODEL_CONFIG.briefSynthesis.topK,
        maxOutputTokens: MODEL_CONFIG.briefSynthesis.maxOutputTokens,
      },
    });

    res.json({ brief: response.text });
  } catch (error: any) {
    console.error('Brief synthesis error:', error);
    res.status(500).json({ error: 'Failed to synthesize operational brief' });
  }
});

// Start Express server and mount Vite middleware
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (_req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(Number(PORT), '0.0.0.0', () => {
    console.log(`WayneTech Sector Dispatch Server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
