# Productivity Agent

A small pipeline that reads your productivity logs, runs several AI agents in order (**Observer → Planner → Simulator → Evaluator**), and optionally an **Adapter** step to compare predictions with what actually happened.

It uses **Google Gemini** for the LLM steps. Default input data is `data/productivity_logs.csv`.

## Setup

1. **Python 3** with these packages (install however you like):

   ```bash
   pip install streamlit pandas pydantic google-genai
   ```

2. **API key** — set your Gemini key in the environment before running:

   **PowerShell**

   ```powershell
   $env:GEMINI_API_KEY = "your-key-here"
   ```

   **macOS / Linux**

   ```bash
   export GEMINI_API_KEY=your-key-here
   ```

## Run it

**Command line** — runs the pipeline and prints the top recommendation. Outputs JSON under `data/`.

```bash
python main.py
```

- Include the Adapter with example metrics: `python main.py --with-adapter`
- Use a different CSV: `python main.py --csv path/to/logs.csv`

**Web UI** — same flow in a browser with a feedback tab for the Adapter.

```bash
streamlit run app.py
```

## What gets saved

Each agent writes to `data/` (for example `observer_output.json`, `planner_output.json`, and so on). The Adapter also updates `adapter_memory.json` when you use it.

---

*Replace sample metrics in `main.py --with-adapter` or use the app’s Feedback tab with your real numbers after trying a strategy.*
