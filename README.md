# LangGraph Supervisor Agent

A small multi-agent system built with **LangGraph** and **Google Gemini**. A supervisor decides which specialist works on each request, specialists use tools to read or change data, and conversation state is persisted in SQLite so sessions can resume.

The project also documents a real bug I found while testing (the agent invented information) and how I fixed it with a code-level rule, a prompt change, and automated tests.

## How it works

```
START → supervisor → investigador → tools_investigador → supervisor
                   → operador     → tools_operador     → supervisor
                   → END (FIN, with the final answer)
```

- **Supervisor:** an LLM with structured output that chooses the next step (`investigador`, `operador`, or `FIN`) and writes the final answer when the task is complete.
- **Investigador:** a specialist with two read-only tools: `consultar_estado_servicio` (service status) and `consultar_ticket` (look up a ticket by ID).
- **Operador:** a specialist with one tool, `crear_ticket`, which opens an incident ticket.
- **State:** messages, the next step, and a step counter that stops runaway loops (`MAX_PASOS`).
- **Persistence:** conversation history is stored per `thread_id` in SQLite (LangGraph checkpointer). Tickets are stored in their own SQLite table, so they survive restarts.

All service data is simulated. The project makes no calls to AWS and has no costs beyond the free tier of the Gemini API.

## The bug: the agent invented information

With a prompt-only approach, the supervisor answered a question about where a ticket was stored with a made-up description of a "management system" and a "control panel". It never called a tool. The model had no data for that question and filled the gap instead of saying so.

I fixed it in two layers:

1. **A rule in code, not in the prompt.** If the user's message mentions a ticket ID or a known service and no tool has returned data for it yet, the graph routes to the investigator before the model decides anything. Prompts are probabilistic; this check is deterministic.
2. **A prompt instruction for response style.** When a tool returns data, the answer must use it, and must say explicitly which information is not available.

I verified the fix by inspecting the graph trace (`stream_mode="updates"`), which shows every node visited and every tool result, not just the final answer.

## Tests

```bash
python -m pytest -v
```

- **Unit tests (6):** no API calls. They cover the routing rule, the ticket tools, and the not-found case.
- **Integration tests (2):** call the real Gemini API and check that the agent does not invent locations for tickets and reports unknown tickets correctly. They are skipped automatically when `GOOGLE_API_KEY` is not set, so CI runs only the unit tests.

Integration tests depend on model output, so they can occasionally fail. A failure means the agent deviated and the trace should be inspected; the assertions are not meant to be relaxed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key-from-google-ai-studio"

python grafo.py demo "Check lambda and open a ticket if it is down."
python grafo.py demo "Where is ticket TCK-0001 stored?"
```

The first command runs the supervisor on the `demo` thread. The second continues the same thread and demonstrates persistence. Use a different thread name (for example `other`) to start a fresh conversation.

## Project structure

```
grafo.py              Graph definition, tools, supervisor and specialists
test_grafo.py         Unit and integration tests
requirements.txt      Dependencies
.github/workflows/    CI: runs unit tests on every push
```

## Limitations and next steps

- Ticket IDs are sequential and derived from the row count, which is fine for a demo but not for concurrent writes.
- The supervisor relies on a single model for routing and for writing the final answer. A stronger evaluation would measure routing accuracy across a labeled set of requests.
- Possible extensions: a third specialist that searches documentation with RAG, and an evaluation script with a fixed dataset.

## Author

Pablo Cueto Gatica, Python backend developer based in Mexico City. Related projects: [enterprise-ai-agent](https://github.com/PabsCueto/enterprise-ai-agent), a RAG agent with LangGraph, evaluation, and a deployed demo.

## License

MIT. See [LICENSE](LICENSE).
# langgraph-supervisor-agent
