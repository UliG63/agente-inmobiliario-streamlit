import os
import sqlite3
import pandas as pd
from difflib import get_close_matches
import streamlit as st
import re 
from typing import List
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from word2number import w2n


# Clave API (puede venir de variable de entorno o input en la app)
if "OPENAI_API_KEY" not in os.environ:
    api_key = st.sidebar.text_input("🔑 Ingresa tu OpenAI API Key", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
else:
    api_key = os.environ["OPENAI_API_KEY"]

if not api_key:
    st.warning("Por favor ingresa tu API Key en la barra lateral para comenzar.")
    st.stop()

# --- RUTAS ABSOLUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.join(BASE_DIR, "real_estate.db")

# --- VERIFICAR EXISTENCIA DE LA BASE DE DATOS ---
# (La app de Streamlit solo lee, nunca escribe)
if not os.path.exists(SQLITE_PATH):
    st.error(f"Error Crítico: No se encuentra el archivo 'real_estate.db'.")
    st.error("Por favor, asegúrese de que el archivo .db esté en el repositorio de GitHub.")
    st.stop()

# --- FUNCIONES ---
# ==========================
# FUNCIONES DE APOYO
# ==========================
def parse_res(rows: List[tuple], cols: List[str], max_rows: int = 50) -> str:
    if not rows:
        return "(sin resultados)"
    header = " | ".join(cols)
    sep = "-|-".join("-" * len(c) for c in cols)
    body = "\n".join(" | ".join(str(x) for x in r) for r in rows[:max_rows])
    note = "" if len(rows) <= max_rows else f"\n… (mostrando primeras {max_rows} filas)"
    return f"{header}\n{sep}\n{body}{note}"

# ==========================
# TOOLS
# ==========================

def sql_schema_tool(db_path: str = SQLITE_PATH) -> str:
    """Devuelve un resumen del esquema (tablas y columnas) de la base SQLite."""
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
            tables = [r[0] for r in cur.fetchall()]
            lines = []
            for t in tables:
                cur.execute(f"PRAGMA table_info({t})")
                cols = ", ".join(f"{c[1]} {c[2]}" for c in cur.fetchall())
                lines.append(f"- {t}: {cols}")
            return "Esquema:\n" + ("\n".join(lines) if lines else "(sin tablas)")
    except Exception as e:
        return f"Error al obtener esquema: {e}"


def sql_query_tool(query: str, db_path: str = SQLITE_PATH, max_rows: int = 50) -> str:
    """Ejecuta UNA consulta SELECT segura sobre SQLite."""
    q = query.strip().replace(";", "")
    if (q.startswith("'") and q.endswith("'")) or (q.startswith('"') and q.endswith('"')):
        q = q[1:-1].strip()

    if ";" in q:
        return "Por seguridad, solo se permite una sentencia por llamada."

    if not re.match(r"(?is)^\s*select\b", q):
        return "Solo se permiten consultas SELECT."

    forbidden = r"\b(insert|update|delete|drop|alter|attach|create|replace|vacuum|pragma)\b"
    if re.search(forbidden, q, flags=re.I):
        return "Se detectó una palabra clave no permitida para lectura segura."

    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(q)
            rows = cur.fetchmany(max_rows)
            cols = [d[0] for d in cur.description] if cur.description else []
            return parse_res(rows, cols, max_rows)
    except Exception as e:
        return f"Error al ejecutar la consulta: {e}"


def text_to_number(text: str) -> float:
    """
    Convierte números escritos en palabras (en español simplificado) a número.
    También maneja expresiones mixtas como 'once mil', 'mil quinientos', etc.
    """
    # Diccionario básico español-inglés para compatibilidad con w2n
    mapping = {
        "uno": "one", "una": "one", "dos": "two", "tres": "three", "cuatro": "four",
        "cinco": "five", "seis": "six", "siete": "seven", "ocho": "eight", "nueve": "nine",
        "diez": "ten", "once": "eleven", "doce": "twelve", "trece": "thirteen", "catorce": "fourteen",
        "quince": "fifteen", "dieciséis": "sixteen", "diecisiete": "seventeen", "dieciocho": "eighteen",
        "diecinueve": "nineteen", "veinte": "twenty", "treinta": "thirty", "cuarenta": "forty",
        "cincuenta": "fifty", "sesenta": "sixty", "setenta": "seventy", "ochenta": "eighty",
        "noventa": "ninety", "cien": "hundred", "ciento": "hundred", "mil": "thousand",
        "millón": "million", "millones": "million", "media": "0.5"
    }

    # Reemplazar palabras españolas por sus equivalentes en inglés para que w2n funcione
    words = text.lower()
    for es, en in mapping.items():
        words = re.sub(rf'\b{es}\b', en, words)

    try:
        return float(w2n.word_to_num(words))
    except Exception:
        return None

def estimate_price_by_zone_tool(text: str, db_path: str = SQLITE_PATH) -> str:
    """
    Estima el precio de una propiedad en base a texto libre.
    Ahora también detecta números escritos con palabras y hectáreas.
    """

    text = text.lower().strip()

    try:
        m2 = None

        # --- Paso 1: Detectar y convertir expresiones con "mil" ---
        mil_match = re.search(r'(\d+(?:[\.,]\d+)?)\s*mil', text)
        if mil_match:
            num_str = mil_match.group(1).replace(',', '.')
            num = float(num_str) * 1000
            text = re.sub(r'(\d+(?:[\.,]\d+)?)\s*mil', str(int(num)), text)

        # --- Paso 2: Detectar hectáreas ---
        ha_match = re.search(
            r'(\d+(?:[\.,]\d+)?)\s*(?:ha|hect[aá]reas?)',
            text, re.IGNORECASE
        )
        if ha_match:
            ha = float(ha_match.group(1).replace(",", "."))
            m2 = ha * 10000

        # --- Paso 3: Detectar número escrito en palabras (por ej. "once mil") ---
        if m2 is None:
            word_number_match = re.search(
                r'([a-záéíóúüñ\s]+)\s*(?:m2|m²|mts2|m\s*cuadrados|mts\s*cuadrados|metros\s*cuadrados|hect[aá]reas?)',
                text, re.IGNORECASE
            )
            if word_number_match:
                word_num = word_number_match.group(1).strip()
                num_val = text_to_number(word_num)
                if num_val:
                    if "hect" in word_number_match.group(0).lower():
                        m2 = num_val * 10000
                    else:
                        m2 = num_val

        # --- Paso 4: Detectar números normales (fallback) ---
        if m2 is None:
            m2_match = re.search(
                r'(\d+(?:[\.,]\d+)?)\s*(?:m2|m²|mts2|m\s*cuadrados|mts\s*cuadrados|metros\s*cuadrados)',
                text, re.IGNORECASE
            )
            if m2_match:
                m2 = float(m2_match.group(1).replace(",", "."))

        if m2 is None:
            return "No pude detectar los metros cuadrados o hectáreas en el texto."

        # --- Paso 5: Detectar nombre de la zona ---
        zone_match = re.search(
            r'\b(?:en\s+la\s+zona\s+de|en\s+zona\s+de|en\s+|zona\s+|barrio\s+)([a-záéíóúüñ\s]+)',
            text, re.IGNORECASE
        )
        if not zone_match:
            return "No pude detectar la zona en el texto."
        zone_name = zone_match.group(1).strip()

        # --- Paso 6: Conectar a la base ---
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute("SELECT name, average_price_per_m2 FROM zones")
            zones = cur.fetchall()

        if not zones:
            return "No hay zonas registradas en la base de datos."

        zone_names = [z[0] for z in zones]
        matches = get_close_matches(zone_name, zone_names, n=1, cutoff=0.5)
        if not matches:
            return f"No se encontró ninguna zona similar a '{zone_name}'."

        matched_zone = matches[0]
        price_per_m2 = next((z[1] for z in zones if z[0] == matched_zone), None)
        if price_per_m2 is None:
            return f"No se pudo obtener el precio por m² para '{matched_zone}'."

        estimated_price = m2 * price_per_m2

        return (
            f"Zona encontrada: {matched_zone}\n"
            f"Precio promedio por m²: {price_per_m2}\n"
            f"Superficie: {m2:,.2f} m²\n"
            f"Precio estimado: ${estimated_price:,.2f}"
        )
    except Exception as e:
        return f"Error al calcular precio estimado: {e}"



def calcular_financiacion_tool(text: str, db_path: str = SQLITE_PATH) -> str:
    """
    Usa esta herramienta cuando el usuario quiera calcular la financiación mensual aproximada de una propiedad o préstamo.

    La herramienta:
    - Identifica automáticamente el monto total (precio o valor del préstamo), la cantidad de cuotas o plazo y la tasa de interés anual a partir del texto del usuario.
    - Calcula la cuota mensual aproximada utilizando una fórmula estándar de préstamos.
    - Devuelve el valor de la cuota mensual y el costo total aproximado del financiamiento.

    Ejemplos de usos típicos:
    - “Quiero financiar una casa de 10.000.000 en 240 cuotas al 7% anual.”
    - “Cuánto pagaría por mes si pido un préstamo de 15 millones a 20 años con una tasa del 8,5%.”
    - “Simulá un crédito hipotecario de 80.000 dólares en 10 años al 5%.”

    Si el texto no incluye monto, cantidad de cuotas (o plazo) y tasa de interés, no puede hacer el cálculo.
    """
    try:
        monto_match = re.search(r'\b(\d+(?:,\d{1,2})?)\s*monto\b', text, re.IGNORECASE)
        if monto_match:
            monto = float(monto_match.group(1).replace(',', '.'))
        else:
            return "No se encontró monto. Formato esperado: '500000,50 monto'"

        cuotas_match = re.search(r'\b(\d+)\s*(?:cuotas|meses)\b', text, re.IGNORECASE)
        if cuotas_match:
            cuotas = int(cuotas_match.group(1))
        else:
            return "No se encontró cantidad de cuotas."

        interes_match = re.search(
            r'(?:inter[eé]s|tasa(?:\s+anual)?|al)?\s*(?:anual\s*)?(?:del\s*)?(\d+(?:[.,]\d+)?)\s*%',
            text, re.IGNORECASE
        )
        if interes_match:
            interes_anual = float(interes_match.group(1).replace(',', '.'))
        else:
            return "No se encontró tasa de interés."

        interes_mensual = interes_anual / 12 / 100
        if interes_mensual == 0:
            pago_mensual = monto / cuotas
        else:
            pago_mensual = monto * (interes_mensual * (1 + interes_mensual)**cuotas) / ((1 + interes_mensual)**cuotas - 1)

        total_pagado = pago_mensual * cuotas
        intereses_totales = total_pagado - monto

        return (
            f"Financiación calculada:\n"
            f"Monto: {monto}\n"
            f"Cuotas: {cuotas}\n"
            f"Interés anual: {interes_anual}%\n"
            f"Pago mensual: {pago_mensual:.2f}\n"
            f"Monto total pagado: {total_pagado:.2f}\n"
            f"Intereses totales: {intereses_totales:.2f}"
        )
    except Exception as e:
        return f"Error en cálculo de financiación: {e}"

# ==========================
# REGISTRO DE TOOLS EN LANGCHAIN
# ==========================
tools = [
    Tool(
        name="SQLSchema",
        func=sql_schema_tool,
        description="Devuelve un resumen del esquema (tablas y columnas) de la base SQLite."
    ),
    Tool(
        name="SQLQuery",
        func=sql_query_tool,
        description="Ejecuta una consulta SELECT segura sobre la base de datos."
    ),
    Tool(
        name="EstimatePriceByZone",
        func=estimate_price_by_zone_tool,
        description="Estima el precio de una propiedad a partir de un texto que indique zona y metros cuadrados."
    ),
    Tool(
        name="CalcularFinanciacion",
        func=calcular_financiacion_tool,
        description="Calcula financiación a partir de monto, cuotas y tasa de interés expresados en texto."
    )
]

# --- CREACIÓN DEL AGENTE ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)


# --------------------------
# Prompt estilo ReAct
# --------------------------
template = """You are a helpful agent that MUST ALWAYS answer in Spanish (Rioplatense).

Conversation so far:
{chat_history}

Follow this format exactly:

Question: the user's question
Thought: briefly explain your plan
Action: the tool to use, exactly one of [{tool_names}]
Action Input: the input for the tool
Observation: the result of the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Final Answer: your concise answer to the user in Spanish

Rules:
- If you choose an Action, DO NOT write 'Final Answer' in the same turn.
- When you choose an Action, respond ONLY with:
  Action: <tool name>
  Action Input: <input>
- Do NOT invent Observations; only write Observation after a tool run.
- If no tool is needed, go directly to Final Answer (in Spanish).

Available tools:
{tools}

Question: {input}
{agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

tools = tools
agent = create_react_agent(llm, tools, prompt)
history = ChatMessageHistory()
with_history = RunnableWithMessageHistory(agent, lambda session_id: history)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Agente Inmobiliario", page_icon="🏡", layout="wide")
st.title("🏡 Asistente Inmobiliario Inteligente")

# --- INICIALIZACIÓN SEGURA DEL HISTORIAL ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNCIÓN PARA AGREGAR MENSAJES ---
def add_message(role: str, content: str):
    """Agrega un mensaje al historial con estructura consistente."""
    st.session_state.messages.append({
        "role": role,
        "content": content
    })

#  --- FUNCIÓN PARA RECONSTRUIR HISTORIAL ---
def build_chat_history() -> str:
    """Convierte el historial de st.session_state en un string para el prompt."""
    history_text = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            history_text += f"Usuario: {msg['content']}\n"
        else:
            history_text += f"Asistente: {msg['content']}\n"
    return history_text.strip()

# --- INPUT DEL USUARIO ---
user_input = st.chat_input("Escribe tu consulta sobre propiedades...")

# --- PROCESAMIENTO DE MENSAJE DEL USUARIO ---
if user_input:
    add_message("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Buscando información..."):
            try:
                response = executor.invoke({
                    "input": user_input,
                    "chat_history": build_chat_history()   
                })
                assistant_response = response.get("output", "(Sin respuesta)")
            except Exception as e:
                assistant_response = f"⚠️ Error procesando la consulta: {e}"

            st.markdown(assistant_response)
            add_message("assistant", assistant_response)

# --- HISTORIAL DE CONVERSACIÓN ---
if st.session_state.messages:
    with st.expander("📝 Historial de conversación"):
        for msg in st.session_state.messages:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.write(f"**{role_icon}**: {msg['content']}")

