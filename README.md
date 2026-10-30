# LangGraph Supervisor Agent

A multi-agent system built with **LangGraph** and **Google Gemini**. A supervisor decides which specialist handles each request. Specialists use tools to read or change data, and conversation state is persisted in SQLite so sessions can resume.

The project also documents the failures I found while building it (an invented answer, a skipped tool call, and an infinite loop), how I diagnosed each one from the graph trace, and how I fixed them with code-level guardrails rather than prompt changes alone.

## How it works

```
START → supervisor → investigador → tools_investigador → supervisor
                   → operador     → tools_operador     → supervisor
                   → END (FIN, with the final answer)
```

- **Supervisor:** an LLM with structured output that chooses the next step (`investigador`, `operador`, or `FIN`) and writes the final answer.
- **Investigador:** reads data with two tools: `consultar_estado_servicio` (service status) and `consultar_ticket` (look up a ticket by ID).
- **Operador:** opens incident tickets with `crear_ticket`.
- **State:** messages, the next step, and a step counter (`MAX_PASOS = 8`) that stops runaway loops.
- **Persistence:** conversation history is stored per `thread_id` (LangGraph SQLite checkpointer). Tickets live in their own SQLite table, so they survive restarts.

All service data is simulated. The project makes no calls to AWS and uses only the free tier of the Gemini API.

## Guardrails

Each guardrail lives in code, not only in the prompt, because prompts are probabilistic and these rules must hold every time.

1. **Consult before answering.** If the user's message mentions a ticket ID or a known service and no tool has returned data for it yet, the graph routes to the investigator before the model decides anything.
2. **No repeated investigation.** Once tool results exist for the current question, the supervisor cannot select the investigator again. If the model tries, the graph finishes with a summary built only from tool output.
3. **Empty responses stop the loop.** If the model returns a message with no text and no tool call, the graph ends with an honest message instead of feeding the empty response back into the conversation.
4. **Step limit.** `MAX_PASOS` caps the number of supervisor decisions.

## Bugs found during development

**1. Invented information.** Asked where a ticket was stored, the supervisor described a "management system" and a "control panel" that did not exist, without calling any tool. The fix combines guardrail 1 with a prompt rule: answers must use only tool output, and must state what is not available.

**2. Infinite loop on empty responses.** For a prompt-injection test, the investigator returned empty messages with no tool calls. Those empty messages were added to the history and sent back to the model, which kept returning empty responses until the step limit. Tracing each node showed that no `ToolMessage` was ever produced. The fix is guardrail 3.

I verified each fix by inspecting the graph trace with `stream_mode="updates"`, which shows every node visited and every tool result, not only the final answer.

## Evaluation

`evals/evaluar.py` runs 13 cases from `evals/casos.json` through the graph and measures two things separately:

- **Routing:** whether the expected specialists were used, in order.
- **Content:** whether the final answer contains the expected text and none of the forbidden text. Forbidden words are matched as whole words.

Current results: **routing 13/13, content 12/13.**

The one failing case is a prompt-injection test (`inyeccion-de-instrucciones`). The agent does not obey the instruction to report that all services are healthy, and the guardrails prevent a loop. However, the model returns no usable data for that input, so the agent does not report the real status of `lambda`. This is a known limitation, documented rather than hidden.

Each evaluation run makes real calls to the Gemini API. It is meant to be run manually, not in CI.

```bash
python evals/evaluar.py
```

## Tests

```bash
python -m pytest -v
```

- **Unit tests (6):** no API calls. They cover the routing rule, the ticket tools, and the not-found case.
- **Integration tests (2):** call the real Gemini API and check that the agent does not invent ticket locations and reports unknown tickets correctly. They are skipped when `GOOGLE_API_KEY` is not set, so CI runs only the unit tests.

Integration tests depend on model output and can occasionally fail. A failure means the agent deviated, and the trace should be inspected rather than the assertion relaxed.

## Setup

Requirements: Python 3.11 or newer and a Google AI Studio API key (free tier).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your-key-from-google-ai-studio"
```

Never commit your key. Use a `.env` file only locally; it is excluded by `.gitignore`.

## Running

```bash
python grafo.py demo "Check lambda and open a ticket if it is down."
python grafo.py demo "Where is ticket TCK-0001 stored?"
```

The first command runs the supervisor on the `demo` thread. The second continues the same thread, which demonstrates persistence. Use another thread name (for example `other`) to start a fresh conversation.

Set `AGENTE_DB` to use a different database file, for example `AGENTE_DB=/tmp/test.db python grafo.py ...`.

## Project structure

```
grafo.py              Graph, tools, supervisor, specialists and guardrails
test_grafo.py         Unit and integration tests
evals/casos.json      Evaluation cases
evals/evaluar.py      Evaluation runner (routing and content)
requirements.txt      Dependencies
.github/workflows/    CI: runs unit tests on every push
```

## Limitations and next steps

- The agent answers only with data from its tools. For questions it has no tool for, such as the current time, it should say so; this is covered by an evaluation case.
- Ticket IDs are sequential and derived from the row count, which is fine for a demo but not for concurrent writes.
- Routing and content are evaluated on 13 hand-written cases. A larger, adversarial set would give a stronger signal.
- Possible next steps: log the reason Gemini returns an empty response, add a third specialist that searches documentation with RAG, and evaluate on a larger dataset.

## Author

Pablo Cueto Gatica, Python backend developer based in Mexico City. Related project: [enterprise-ai-agent](https://github.com/PabsCueto/enterprise-ai-agent), a RAG agent with LangGraph, evaluation, and a deployed demo.

## License

MIT. See [LICENSE](LICENSE).
