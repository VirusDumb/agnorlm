# Recursive Language Models (RLM) with Agno

**Based on:** [Recursive Language Models](https://arxiv.org/abs/2512.24601) — Zhang, Kraska, Khattab (MIT CSAIL, 2025)

## What is an RLM?

An RLM is an inference strategy that lets LLMs process inputs **far beyond their context window** — up to 100x larger — by treating the prompt as data in a Python REPL rather than feeding it directly into the model.

Think of it like **out-of-core algorithms**: your computer has 16GB of RAM but can process a 1TB dataset by reading chunks from disk as needed. The RLM does the same with text: the LLM's context window is the "RAM", and the full prompt sits in the REPL environment like a "disk".

## Architecture
```
User Question + Long Prompt
         │
         ▼
┌─────────────────────────────────┐
│     RLM Root Agent              │
│  (Claude Sonnet / GPT-5)        │
│                                 │
│  1. Sees: len(prompt), peek     │
│  2. Writes Python to chunk      │
│  3. Calls sub-LM on chunks      │
│  4. Aggregates with Python      │
│                                 │
│  Tools:                         │
│  ├── PythonTools (REPL)         │
│  └── SubLMTool (recursive)      │
│       └── Sub-Agent             │
│           (Claude Haiku)        │
└─────────────────────────────────┘
         │
         ▼
      Answer
```

## Quick Start
```python
from rlm import create_rlm_agent, run_rlm

# Create the RLM agent
agent = create_rlm_agent(
    root_model="claude-sonnet-4-5-20250929",  # The "brain"
    sub_model="claude-haiku-4-5-20251001",    # Cheap sub-calls
)

# Load your massive text
with open("giant_document.txt") as f:
    text = f.read()  # Could be millions of tokens!

# Ask a question
answer = run_rlm(
    agent=agent,
    prompt=text,
    question="What are the key findings about X?",
)
print(answer)
```

## Using with Ollama (Local Inference)
```python
from agno.models.ollama import Ollama
from agno.tools.python import PythonTools
from rlm import SubLMTool, RLM_SYSTEM_PROMPT

class OllamaSubLMTool(SubLMTool):
    def _get_sub_model(self):
        return Ollama(id="qwen2.5:3b")

agent = Agent(
    name="RLM-Local",
    model=Ollama(id="qwen2.5:14b"),
    tools=[PythonTools(), OllamaSubLMTool()],
    instructions=[RLM_SYSTEM_PROMPT],
)
```

## Task Type Guidelines

| Task Type | Example | deep_recursion | Strategy |
|-----------|---------|----------------|----------|
| Needle-in-Haystack | Find a specific fact | `False` | Binary search or keyword scan |
| Multi-hop QA | Cross-reference docs | `False` | Scan headers → deep-read relevant |
| Dense Aggregation | Classify every line | `True` | Chunk → sub-LM each → aggregate |
| Pairwise Reasoning | Compare all pairs | `True` | Strategic pair sampling |
| Code Understanding | Find bugs in repo | `False` | Parse file structure → analyze relevant |

## Key Design Decisions

1. **Root model vs Sub model**: Use a strong model (Sonnet) as the "planner" and a cheap model (Haiku) for chunk processing. The paper found GPT-5 + GPT-5-mini was optimal.

2. **When to use sub-LM vs Python**: Use Python for mechanical tasks (counting, regex, sorting). Use sub-LM only when semantic understanding is needed (classification, summarization, reasoning).

3. **Deep recursion**: Only enable for information-dense tasks where even chunks need further decomposition. For simpler tasks, the non-recursive ablation actually performed comparably.

4. **Chunk size**: Keep sub-prompts under ~4000 tokens. The agent decides chunking strategy based on the task.

## Dependencies
```
pip install agno anthropic  # or openai, ollama
```

## Paper Results Summary

RLMs outperformed base LLMs and common scaffolds (summary agents, CodeAct+BM25) by up to 2x on long-context benchmarks, while maintaining comparable or lower API costs. They successfully handled inputs up to 10M+ tokens — 100x beyond model context windows.
