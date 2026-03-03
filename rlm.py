"""
Recursive Language Model (RLM) Implementation using Agno
Based on: arxiv.org/abs/2512.24601 (Zhang, Kraska, Khattab - MIT CSAIL, 2025)
"""

import json, os, textwrap
from typing import Optional

from agno.agent import Agent
from agno.models.anthropic import Claude
from agno.models.openai import OpenAIChat
from agno.tools.python import PythonTools
from agno.tools import Toolkit


# =============================================================================
# 1. SubLM Tool
# =============================================================================

class SubLMTool(Toolkit):
    def __init__(self, sub_model=None, max_recursion_depth=3, name="sub_lm_tools"):
        super().__init__(name=name)
        self.sub_model = sub_model
        self.max_recursion_depth = max_recursion_depth
        self._current_depth = 0
        self.register(self.query_sub_lm)

    def query_sub_lm(self, sub_prompt: str, task: str) -> str:
        """Query a sub-language-model with a text snippet and a task.

        Args:
            sub_prompt: Text snippet for the sub-LM (few thousand tokens max).
            task: What the sub-LM should do. Be specific!

        Returns:
            The sub-LM response as a string.
        """
        if self._current_depth >= self.max_recursion_depth:
            return "[MAX RECURSION DEPTH] Process with Python instead."
        self._current_depth += 1
        try:
            sub_agent = Agent(
                name="RLM-SubAgent",
                model=self._get_sub_model(),
                instructions=[
                    "You are a focused sub-agent. Complete the task precisely.",
                    "Do NOT add preamble. Return ONLY the requested output.",
                ],
                markdown=False,
            )
            combined = f"## Task\n{task}\n\n## Text to process\n{sub_prompt}"
            response = sub_agent.run(combined)
            if response and response.content:
                return response.content
            return "[SubLM returned empty response]"
        except Exception as e:
            return f"[SubLM Error: {str(e)}]"
        finally:
            self._current_depth -= 1

    def _get_sub_model(self):
        if self.sub_model:
            if "claude" in self.sub_model.lower():
                return Claude(id=self.sub_model)
            elif "gpt" in self.sub_model.lower():
                return OpenAIChat(id=self.sub_model)
        return Claude(id="claude-haiku-4-5-20251001")


# =============================================================================
# 2. Recursive SubLM Tool
# =============================================================================

class RecursiveSubLMTool(Toolkit):
    def __init__(self, root_model=None, sub_model=None, max_recursion_depth=3,
                 name="recursive_sub_lm_tools"):
        super().__init__(name=name)
        self.root_model = root_model
        self.sub_model = sub_model
        self.max_recursion_depth = max_recursion_depth
        self._current_depth = 0
        self.register(self.query_sub_lm_recursive)

    def query_sub_lm_recursive(self, sub_prompt: str, task: str) -> str:
        """Query a sub-LM that itself has Python and sub-LM capabilities.

        Args:
            sub_prompt: The text snippet for the sub-agent.
            task: What the sub-agent should accomplish.

        Returns:
            The sub-agent response.
        """
        if self._current_depth >= self.max_recursion_depth:
            return "[MAX RECURSION DEPTH]"
        self._current_depth += 1
        try:
            sub_python = PythonTools()  # ONLY accepts: base_dir, safe_globals, safe_locals
            nested = SubLMTool(
                sub_model=self.sub_model,
                max_recursion_depth=self.max_recursion_depth - self._current_depth,
            )
            model = self._get_model(self.sub_model or self.root_model)
            sub_agent = Agent(
                name=f"RLM-SubAgent-D{self._current_depth}",
                model=model,
                tools=[sub_python, nested],
                instructions=[
                    "You are a recursive sub-agent with a Python REPL.",
                    "Be efficient. Use sub-LM only for semantic understanding.",
                ],
                markdown=False,
            )
            response = sub_agent.run(f"## Task\n{task}\n\n## Text\n{sub_prompt}")
            if response and response.content:
                return response.content
            return "[RecursiveSubLM returned empty response]"
        except Exception as e:
            return f"[RecursiveSubLM Error: {str(e)}]"
        finally:
            self._current_depth -= 1

    def _get_model(self, model_id):
        if model_id and "claude" in model_id.lower():
            return Claude(id=model_id)
        elif model_id and "gpt" in model_id.lower():
            return OpenAIChat(id=model_id)
        return Claude(id="claude-haiku-4-5-20251001")


# =============================================================================
# 3. System Prompt
# =============================================================================

RLM_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a Recursive Language Model (RLM). You solve tasks over arbitrarily
    long inputs by treating the input as data in your Python environment - NOT
    by trying to read it all at once in your context window.

    ## Your Environment

    You have a Python REPL (via run_python_code or save_to_file_and_run) and
    a query_sub_lm tool.

    The user input is stored in a file on disk. Your FIRST step should always
    be to load it into a Python variable using the file path provided.

    ## Your Strategy

    1. **Load the input**: Read the file into a variable. Check its length.
       Peek at the beginning, middle, and end.
    2. **Decompose programmatically**: Split into manageable chunks with Python.
    3. **Process chunks**: Use query_sub_lm(chunk, task) for semantic work.
       Use Python directly for mechanical ops (counting, regex, filtering).
    4. **Aggregate results**: Combine sub-results with Python code.
    5. **Recurse when needed**: If a chunk is still too large, split further.

    ## Guidelines

    - NEVER print the entire input.
    - Be EFFICIENT with sub-LM calls. Python for mechanical tasks.
    - Keep sub-prompts under ~4000 tokens each.
    - Store intermediate results in variables and aggregate at the end.
""")


# =============================================================================
# 4. Agent Factory
# =============================================================================

def create_rlm_agent(
    root_model="claude-sonnet-4-5-20250929",
    sub_model="claude-haiku-4-5-20251001",
    max_recursion_depth=3,
    deep_recursion=False,
    additional_instructions=None,
    base_dir=None,
):
    if "claude" in root_model.lower():
        model = Claude(id=root_model)
    elif "gpt" in root_model.lower():
        model = OpenAIChat(id=root_model)
    else:
        model = Claude(id=root_model)

    instructions = [RLM_SYSTEM_PROMPT]
    if additional_instructions:
        instructions.extend(additional_instructions)

    # PythonTools: ONLY base_dir, safe_globals, safe_locals
    python_tools = PythonTools(base_dir=base_dir)

    if deep_recursion:
        sub_lm_tool = RecursiveSubLMTool(
            root_model=root_model, sub_model=sub_model,
            max_recursion_depth=max_recursion_depth,
        )
    else:
        sub_lm_tool = SubLMTool(
            sub_model=sub_model, max_recursion_depth=max_recursion_depth,
        )

    return Agent(
        name="RLM-Agent", model=model,
        tools=[python_tools, sub_lm_tool],
        instructions=instructions,
        markdown=True,
    )


# =============================================================================
# 5. Runner
# =============================================================================

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_rlm(agent, prompt, question, verbose=True):
    """Write prompt to temp file, give agent metadata + path, let it work."""
    tmp_file = os.path.join(SCRIPT_DIR, "rlm_prompt.txt")
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    first_200 = prompt[:200].replace('\n', ' ')
    last_200 = prompt[-200:].replace('\n', ' ')

    setup_message = (
        "## Setup\n\n"
        "A large text input has been saved to disk. Load it like this:\n\n"
        "```python\n"
        f'with open(r"{tmp_file}", "r", encoding="utf-8") as f:\n'
        "    prompt = f.read()\n"
        'print(f"Loaded {len(prompt)} characters")\n'
        "```\n\n"
        "Key facts:\n"
        f"- **Length**: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)\n"
        f'- **First 200 chars**: "{first_200}"\n'
        f'- **Last 200 chars**: "{last_200}"\n\n'
        f"## Your Task\n\n{question}\n\n"
        "## Remember\n\n"
        "- First load the file using the code above.\n"
        "- Use len(prompt), slicing, and Python string ops.\n"
        "- Use query_sub_lm(chunk, task) for semantic understanding.\n"
        "- Use Python directly for mechanical operations.\n"
        "- Build your answer incrementally.\n"
    )

    if verbose:
        response = agent.print_response(setup_message, stream=True)
    else:
        response = agent.run(setup_message)

    try:
        os.remove(tmp_file)
    except OSError:
        pass

    if response and response.content:
        return response.content
    return ""


# =============================================================================
# 6. Convenience
# =============================================================================

def rlm_summarize(text, model="claude-sonnet-4-5-20250929"):
    agent = create_rlm_agent(root_model=model)
    return run_rlm(agent, text, "Comprehensive summary covering all major topics.")

def rlm_search(text, query, model="claude-sonnet-4-5-20250929"):
    agent = create_rlm_agent(root_model=model)
    return run_rlm(agent, text, f"Find and extract: {query}")

def rlm_aggregate(text, task, model="claude-sonnet-4-5-20250929"):
    agent = create_rlm_agent(root_model=model, deep_recursion=True)
    return run_rlm(agent, text, task)


# =============================================================================
# 7. Demo
# =============================================================================

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("RLM Demo: Needle in a Haystack")
    print("=" * 60)

    random.seed(42)
    filler = [
        "The quarterly earnings report showed steady growth in the consumer segment.",
        "Weather patterns in the Pacific Northwest remained consistent with forecasts.",
        "The committee discussed improvements to the municipal water treatment facility.",
        "Research in quantum computing continues to yield promising theoretical results.",
        "The new highway interchange project is expected to reduce commute times by 15%.",
        "Library circulation numbers have increased since the renovation was completed.",
        "The agricultural department released updated guidelines for crop rotation.",
        "Maritime shipping routes were adjusted due to seasonal ice formation patterns.",
        "The archaeological survey uncovered pottery fragments dating to the 3rd century.",
        "Solar panel efficiency improvements have made residential installation more viable.",
    ]

    lines = [f"[Doc {i+1}] {random.choice(filler)}" for i in range(5000)]
    needle_pos = random.randint(2000, 3000)
    lines.insert(needle_pos,
        "[Doc SPECIAL] The secret activation code for Project Chimera is: PHOENIX-7742-DELTA.")

    haystack = "\n".join(lines)
    print(f"Haystack: {len(haystack):,} chars ({len(lines)} lines)")
    print(f"Needle at line {needle_pos + 1}\n")

    rlm_agent = create_rlm_agent(
        root_model="claude-sonnet-4-5-20250929",
        sub_model="claude-haiku-4-5-20251001",
    )

    answer = run_rlm(
        agent=rlm_agent, prompt=haystack,
        question="Find the secret activation code for Project Chimera. What is it?",
    )

    print(f"\n{'='*60}\nAnswer: {answer}\n{'='*60}")
