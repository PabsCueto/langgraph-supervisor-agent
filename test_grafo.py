"""
Pruebas del agente supervisor.

Unitarias: no llaman a la API, verifican reglas deterministas y herramientas.
Integración: llaman a Gemini real; se omiten si no hay GOOGLE_API_KEY.

Ejecutar:  pytest -v
"""
import os
import tempfile

# Base temporal para no tocar tu agente.db real, y clave ficticia para importar sin red.
os.environ["AGENTE_DB"] = os.path.join(tempfile.mkdtemp(), "pruebas.db")
os.environ.setdefault("GOOGLE_API_KEY", "clave-ficticia-para-pruebas")

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import grafo

CLAVE_REAL = os.environ.get("GOOGLE_API_KEY", "")
HAY_CLAVE_REAL = CLAVE_REAL not in ("", "clave-ficticia-para-pruebas")


# ---------- Unitarias: reglas en código ----------
def test_menciona_ticket_sin_consulta_requiere_consulta():
    assert grafo.requiere_consulta([HumanMessage("¿Dónde está TCK-0001?")]) is True


def test_menciona_servicio_sin_consulta_requiere_consulta():
    assert grafo.requiere_consulta([HumanMessage("Revisa lambda, por favor")]) is True


def test_sin_mencion_no_requiere_consulta():
    assert grafo.requiere_consulta([HumanMessage("Hola, ¿cómo estás?")]) is False


def test_con_resultado_de_herramienta_ya_no_requiere_consulta():
    mensajes = [
        HumanMessage("¿Qué pasó con TCK-0001?"),
        AIMessage(content="", tool_calls=[{"name": "consultar_ticket", "args": {"ticket_id": "TCK-0001"}, "id": "1"}]),
        ToolMessage(content="ticket TCK-0001 | título: x", tool_call_id="1"),
    ]
    assert grafo.requiere_consulta(mensajes) is False


def test_consultar_ticket_inexistente_lo_reporta():
    resultado = grafo.consultar_ticket.invoke({"ticket_id": "TCK-9999"})
    assert "no existe" in resultado


def test_ticket_creado_se_puede_consultar():
    creado = grafo.crear_ticket.invoke({"titulo": "Prueba unitaria", "descripcion": "Descripción de prueba"})
    ticket_id = creado.split()[1]  # "ticket TCK-XXXX creado"
    consulta = grafo.consultar_ticket.invoke({"ticket_id": ticket_id})
    assert "Prueba unitaria" in consulta
    assert "Descripción de prueba" in consulta


# ---------- Integración: modelo real ----------
PALABRAS_PROHIBIDAS = ["panel", "sistema de gestión", "ubicado en", "servidor", "base de datos de"]


@pytest.mark.skipif(not HAY_CLAVE_REAL, reason="requiere GOOGLE_API_KEY real")
def test_integracion_no_inventa_ubicacion_de_ticket():
    creado = grafo.crear_ticket.invoke({"titulo": "Lambda degradado", "descripcion": "Latencia alta"})
    ticket_id = creado.split()[1]
    config = {"configurable": {"thread_id": "pytest-ubicacion"}}
    entrada = {
        "messages": [HumanMessage(f"¿Dónde se almacena el ticket {ticket_id}?")],
        "siguiente": "",
        "pasos": 0,
    }
    resultado = grafo.app.invoke(entrada, config)
    respuesta = resultado["messages"][-1].content.lower()
    for palabra in PALABRAS_PROHIBIDAS:
        assert palabra not in respuesta, f"la respuesta menciona '{palabra}', posible alucinación"


@pytest.mark.skipif(not HAY_CLAVE_REAL, reason="requiere GOOGLE_API_KEY real")
def test_integracion_ticket_inexistente_se_reporta():
    config = {"configurable": {"thread_id": "pytest-inexistente"}}
    entrada = {
        "messages": [HumanMessage("¿Qué pasó con el ticket TCK-9999?")],
        "siguiente": "",
        "pasos": 0,
    }
    resultado = grafo.app.invoke(entrada, config)
    respuesta = resultado["messages"][-1].content.lower()
    assert "no existe" in respuesta or "no tengo" in respuesta
