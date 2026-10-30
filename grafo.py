"""
Agente supervisor con LangGraph.

Un supervisor decide qué especialista trabaja:
- investigador: consulta estados de servicios y tickets (solo con herramientas).
- operador: abre tickets de incidente.
Cada especialista devuelve el control al supervisor, que decide si la tarea terminó.

Persistencia:
- Historial de conversación por thread_id (tabla de checkpoints de LangGraph).
- Tickets en una tabla propia de SQLite, así los datos de negocio sobreviven entre ejecuciones.
"""
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_PASOS = 8  # límite de seguridad para evitar ciclos infinitos
RUTA_DB = os.environ.get("AGENTE_DB", "agente.db")

# ---------- Base de datos de tickets (SQLite) ----------
datos = sqlite3.connect(RUTA_DB, check_same_thread=False)
datos.execute(
    """CREATE TABLE IF NOT EXISTS tickets (
        id TEXT PRIMARY KEY,
        titulo TEXT NOT NULL,
        descripcion TEXT NOT NULL,
        creado TEXT NOT NULL
    )"""
)
datos.commit()

# Estados simulados de servicios (no tocan AWS ni cobran nada)
ESTADOS = {"ecs": "operativo", "dynamodb": "operativo", "lambda": "degradado (latencia alta)"}


# ---------- Herramientas ----------
@tool
def consultar_estado_servicio(servicio: str) -> str:
    """Devuelve el estado actual de un servicio de AWS, por ejemplo lambda."""
    return ESTADOS.get(servicio.lower(), f"servicio '{servicio}' no encontrado")


@tool
def crear_ticket(titulo: str, descripcion: str) -> str:
    """Abre un ticket de incidente con título y descripción. Devuelve su ID."""
    (total,) = datos.execute("SELECT COUNT(*) FROM tickets").fetchone()
    ticket_id = f"TCK-{total + 1:04d}"
    datos.execute(
        "INSERT INTO tickets (id, titulo, descripcion, creado) VALUES (?, ?, ?, ?)",
        (ticket_id, titulo, descripcion, datetime.now(timezone.utc).isoformat()),
    )
    datos.commit()
    return f"ticket {ticket_id} creado"


@tool
def consultar_ticket(ticket_id: str) -> str:
    """Busca un ticket por su ID, por ejemplo TCK-0001, en la base de datos de tickets."""
    fila = datos.execute(
        "SELECT id, titulo, descripcion, creado FROM tickets WHERE id = ?",
        (ticket_id.strip().upper(),),
    ).fetchone()
    if fila is None:
        return f"no existe ningún ticket con ID {ticket_id}"
    return f"ticket {fila[0]} | título: {fila[1]} | descripción: {fila[2]} | creado: {fila[3]}"


# ---------- Modelo ----------
llm = ChatGoogleGenerativeAI(model=MODEL, temperature=0)
llm_investigador = llm.bind_tools([consultar_estado_servicio, consultar_ticket])
llm_operador = llm.bind_tools([crear_ticket])


# ---------- Estado compartido ----------
class Decision(BaseModel):
    siguiente: Literal["investigador", "operador", "FIN"]
    respuesta: str = ""  # solo se llena cuando siguiente es FIN


class Estado(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    siguiente: str
    pasos: int


# ---------- Nodos ----------
PROMPT_SUPERVISOR = """Eres el supervisor de un equipo de operaciones AWS.
Especialistas:
- investigador: consulta el estado de servicios y busca tickets por su ID. Es el único que puede obtener esos datos.
- operador: abre tickets de incidente.

Decide el siguiente paso:
- Si el usuario pide un dato (estado de un servicio o detalles de un ticket) que todavía no aparece en los mensajes, elige "investigador".
- Si hay un servicio caído confirmado y el usuario pidió abrir un ticket, elige "operador".
- Si la tarea está completa, elige "FIN" y escribe en "respuesta" el resumen para el usuario.

Reglas para "respuesta":
- Usa solo datos que aparezcan en los resultados de las herramientas dentro de los mensajes.
- Si una herramienta devolvió datos, úsalos en la respuesta aunque falte otro dato. Por ejemplo, si el ticket existe, cuenta sus datos y aclara solo lo que no está disponible.
- Si el investigador consultó y la herramienta no encontró el dato, dilo claramente, por ejemplo: "No tengo esa información".
- Nunca describas sistemas, paneles, ubicaciones ni procesos que no aparezcan en los datos."""


PATRON_TICKET = re.compile(r"TCK-\d{4}", re.IGNORECASE)


def requiere_consulta(messages) -> bool:
    """Regla en código, no en el prompt: si la pregunta menciona un ticket o un servicio
    y aún no hay resultados de herramientas para ella, hay que consultar antes de responder."""
    indices_humanos = [i for i, m in enumerate(messages) if m.type == "human"]
    if not indices_humanos:
        return False
    i = indices_humanos[-1]
    texto = messages[i].content.lower()
    menciona = PATRON_TICKET.search(messages[i].content) or any(s in texto for s in ESTADOS)
    hubo_herramienta = any(m.type == "tool" for m in messages[i + 1:])
    return bool(menciona) and not hubo_herramienta


def hay_resultados_recientes(messages) -> bool:
    """True si ya hubo resultados de herramientas después del último mensaje del usuario."""
    indices = [i for i, m in enumerate(messages) if m.type == "human"]
    if not indices:
        return False
    return any(m.type == "tool" for m in messages[indices[-1] + 1:])


def resumen_de_resultados(messages) -> str:
    """Respuesta de respaldo que usa solo los datos devueltos por las herramientas."""
    indices = [i for i, m in enumerate(messages) if m.type == "human"]
    resultados = [m.content for m in messages[indices[-1] + 1:] if m.type == "tool"]
    return "Resultados: " + "; ".join(resultados)


def supervisor(state: Estado):
    if state["pasos"] >= MAX_PASOS:
        return {
            "siguiente": "FIN",
            "messages": [AIMessage(content="Me quedé sin pasos antes de terminar.")],
        }
    # Guardia: una respuesta vacía del modelo (sin texto ni llamada a herramienta)
    # no debe alimentar otra vuelta. Terminamos con un mensaje honesto.
    ultimo = state["messages"][-1]
    if ultimo.type == "ai" and not getattr(ultimo, "tool_calls", None) and not ultimo.content:
        return {
            "siguiente": "FIN",
            "messages": [AIMessage(content="No pude completar la consulta con las herramientas. Intenta de nuevo con una pregunta más concreta.")],
        }

    if requiere_consulta(state["messages"]):
        return {"siguiente": "investigador", "pasos": state["pasos"] + 1}
    consulto = hay_resultados_recientes(state["messages"])
    prompt = PROMPT_SUPERVISOR
    if consulto:
        prompt += (
            "\n\nYa tienes resultados de herramientas para la pregunta actual. "
            "No pidas más investigación: elige 'operador' si hay que abrir un ticket, "
            "o 'FIN' con la respuesta basada en esos resultados."
        )
    decision = llm.with_structured_output(Decision).invoke(
        [SystemMessage(content=prompt)] + state["messages"]
    )
    siguiente, respuesta = decision.siguiente, decision.respuesta

    # Guardrail en código: con datos ya obtenidos, no se permite otra vuelta de investigación.
    if consulto and siguiente == "investigador":
        siguiente = "FIN"
        respuesta = resumen_de_resultados(state["messages"])

    update = {"siguiente": siguiente, "pasos": state["pasos"] + 1}
    if siguiente == "FIN":
        update["messages"] = [AIMessage(content=respuesta)]
    return update


def investigador(state: Estado):
    msg = llm_investigador.invoke(
        [SystemMessage(content=(
            "Eres el investigador. Obtén los datos solo con las herramientas: "
            "estados de servicios o tickets por ID. No respondas de memoria y no abras tickets."
        ))]
        + state["messages"]
    )
    return {"messages": [msg]}


def operador(state: Estado):
    msg = llm_operador.invoke(
        [SystemMessage(content=(
            "Eres el operador. Abre tickets de incidente con la herramienta, "
            "con título y descripción claros basados en los datos de la conversación."
        ))]
        + state["messages"]
    )
    return {"messages": [msg]}


# ---------- Grafo ----------
grafo = StateGraph(Estado)
grafo.add_node("supervisor", supervisor)
grafo.add_node("investigador", investigador)
grafo.add_node("tools_investigador", ToolNode([consultar_estado_servicio, consultar_ticket]))
grafo.add_node("operador", operador)
grafo.add_node("tools_operador", ToolNode([crear_ticket]))

grafo.add_edge(START, "supervisor")
grafo.add_conditional_edges(
    "supervisor",
    lambda state: state["siguiente"],
    {"investigador": "investigador", "operador": "operador", "FIN": END},
)
grafo.add_edge("investigador", "tools_investigador")
grafo.add_edge("tools_investigador", "supervisor")
grafo.add_edge("operador", "tools_operador")
grafo.add_edge("tools_operador", "supervisor")

# ---------- Persistencia del historial ----------
app = grafo.compile(checkpointer=SqliteSaver(sqlite3.connect(RUTA_DB, check_same_thread=False)))


if __name__ == "__main__":
    hilo = sys.argv[1] if len(sys.argv) > 1 else "demo"
    pregunta = sys.argv[2] if len(sys.argv) > 2 else (
        "Revisa lambda y ecs, y abre un ticket solo por lo que esté caído."
    )
    config = {"configurable": {"thread_id": hilo}}
    entrada = {"messages": [HumanMessage(content=pregunta)], "siguiente": "", "pasos": 0}

    # stream_mode="updates" muestra cada nodo que el grafo visita, en orden.
    respuesta_final = ""
    for evento in app.stream(entrada, config, stream_mode="updates"):
        for nodo, salida in evento.items():
            print(f"  -> nodo: {nodo}")
            if "siguiente" in salida:
                print(f"     supervisor decide: {salida['siguiente']}")
            if "messages" in salida and salida["messages"]:
                ultimo = salida["messages"][-1]
                if ultimo.type == "tool":
                    print(f"     resultado de herramienta: {ultimo.content}")
                if nodo == "supervisor" and salida.get("siguiente") == "FIN":
                    respuesta_final = ultimo.content

    print("\nRespuesta:")
    print(respuesta_final)
    print("\nTickets en la base de datos:")
    for fila in datos.execute("SELECT id, titulo FROM tickets ORDER BY id"):
        print(f"  {fila[0]}: {fila[1]}")
