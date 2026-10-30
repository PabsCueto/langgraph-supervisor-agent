"""
Evaluación del agente supervisor.

Ejecuta cada caso de casos.json en el grafo y mide dos cosas:
1. Ruteo: ¿se usaron los especialistas esperados, en ese orden?
2. Contenido: ¿la respuesta final contiene lo esperado y no contiene lo prohibido?

Las palabras prohibidas se buscan como palabras completas (no como subcadenas),
para evitar falsos positivos como "am" dentro de "amigo".

Cada corrida usa una base temporal nueva, así que no toca agente.db.
Las llamadas consumen la capa gratuita de Gemini: ejecútalo a mano, no en CI.

Uso (desde la carpeta del proyecto):  python evals/evaluar.py
"""
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

# Base temporal limpia para cada corrida. Debe definirse antes de importar grafo.
os.environ["AGENTE_DB"] = os.path.join(tempfile.mkdtemp(), "evaluacion.db")

from langchain_core.messages import HumanMessage  # noqa: E402

import grafo  # noqa: E402

ESPECIALISTAS = {"investigador", "operador"}
REINTENTOS = 3
ERRORES_TRANSITORIOS = ("429", "500", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE")


def ejecutar(pregunta: str):
    """Corre el grafo y devuelve (ruta de especialistas, respuesta final)."""
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    entrada = {"messages": [HumanMessage(content=pregunta)], "siguiente": "", "pasos": 0}
    ruta, respuesta = [], ""
    for evento in grafo.app.stream(entrada, config, stream_mode="updates"):
        for nodo, salida in evento.items():
            if nodo in ESPECIALISTAS and (not ruta or ruta[-1] != nodo):
                ruta.append(nodo)
            if nodo == "supervisor" and salida.get("siguiente") == "FIN":
                respuesta = salida["messages"][-1].content
    return ruta, respuesta


def ejecutar_con_reintentos(pregunta: str):
    """Devuelve ((ruta, respuesta), None) o (None, motivo). Un fallo de la API no cuenta contra el agente."""
    for intento in range(1, REINTENTOS + 1):
        try:
            return ejecutar(pregunta), None
        except Exception as error:
            texto = str(error)
            transitorio = any(codigo in texto for codigo in ERRORES_TRANSITORIOS)
            if transitorio and intento < REINTENTOS:
                time.sleep(5 * 2 ** intento)
                continue
            if transitorio:
                return None, "no evaluado: la API está saturada"
            return None, f"no evaluado: {texto[:100]}"


def contiene_palabra(texto: str, palabra: str) -> bool:
    """Busca la palabra completa, sin distinguir mayúsculas."""
    return re.search(rf"\b{re.escape(palabra)}\b", texto, flags=re.IGNORECASE) is not None


def main():
    if not os.environ.get("GOOGLE_API_KEY"):
        sys.exit("Falta GOOGLE_API_KEY en el entorno.")

    casos = json.loads((Path(__file__).parent / "casos.json").read_text(encoding="utf-8"))
    evaluados = ok_ruta = ok_contenido = 0

    print(f"{'caso':<28}{'ruta':<8}{'contenido':<11}detalle")
    print("-" * 80)
    for caso in casos:
        resultado, motivo = ejecutar_con_reintentos(caso["pregunta"])
        if motivo:
            print(f"{caso['id']:<28}{'—':<8}{'—':<11}{motivo}")
            continue

        ruta_real, respuesta = resultado
        ruta_ok = ruta_real == caso["ruta"]
        contenido_ok = (
            all(contiene_palabra(respuesta, p) or p.lower() in respuesta.lower() for p in caso["debe_contener"])
            and not any(contiene_palabra(respuesta, p) for p in caso["no_debe_contener"])
        )

        evaluados += 1
        ok_ruta += ruta_ok
        ok_contenido += contenido_ok

        detalle = ""
        if not ruta_ok:
            detalle += f"ruta real {ruta_real}, esperada {caso['ruta']}. "
        if not contenido_ok:
            detalle += "contenido no cumple. "
        if caso.get("nota"):
            detalle += f"[{caso['nota']}]"

        marca = lambda ok: "✔" if ok else "✘"
        print(f"{caso['id']:<28}{marca(ruta_ok):<8}{marca(contenido_ok):<11}{detalle.strip()}")

    print("-" * 80)
    print(f"Casos evaluados: {evaluados}/{len(casos)}")
    if evaluados:
        print(f"Ruteo correcto:     {ok_ruta}/{evaluados} ({100 * ok_ruta / evaluados:.0f}%)")
        print(f"Contenido correcto: {ok_contenido}/{evaluados} ({100 * ok_contenido / evaluados:.0f}%)")


if __name__ == "__main__":
    main()
