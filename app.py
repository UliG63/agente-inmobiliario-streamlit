import os
import sqlite3
import pandas as pd
from difflib import get_close_matches
import streamlit as st

from langchain-openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Agente Inmobiliario", page_icon="🏡", layout="wide")
st.title("🏡 Asistente Inmobiliario Inteligente")

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

# --- CARGA DE DATOS ---
SQLITE_PATH = "real_estate.db"
zones_path = "zones.csv"
properties_path = "properties.csv"

# Si la BD no existe, la creamos
if not os.path.exists(SQLITE_PATH):
    zones_df = pd.read_csv(zones_path)
    properties_df = pd.read_csv(properties_path)
    conn = sqlite3.connect(SQLITE_PATH)
    zones_df.to_sql('zones', conn, if_exists='replace', index=False)
    properties_df.to_sql('properties', conn, if_exists='replace', index=False)
    conn.close()

# --- FUNCIONES ---
@tool
def buscar_propiedad(zona: str) -> str:
    """Busca propiedades por zona (coincidencias aproximadas)."""
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT zone FROM properties")
    zonas_db = [row[0] for row in cursor.fetchall()]
    match = get_close_matches(zona, zonas_db, n=1, cutoff=0.6)
    if match:
        cursor.execute("SELECT * FROM properties WHERE zone = ?", (match[0],))
        rows = cursor.fetchall()
        conn.close()
        if rows:
            return f"Propiedades encontradas en {match[0]}:\n" + "\n".join([str(r) for r in rows])
        else:
            return f"No hay propiedades disponibles actualmente en {match[0]}."
    else:
        conn.close()
        return f"No se encontró ninguna zona similar a '{zona}'."

@tool
def listar_zonas() -> str:
    """Lista todas las zonas disponibles."""
    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT zone FROM properties")
    zonas = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ", ".join(zonas)

# --- CREACIÓN DEL AGENTE ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

prompt = PromptTemplate(
    template=(
        "Eres un asistente inmobiliario. "
        "Puedes listar zonas disponibles con listar_zonas y buscar propiedades con buscar_propiedad. "
        "Responde de forma clara y amigable.\n\n"
        "Pregunta del usuario:\n{input}"
    ),
    input_variables=["input"]
)

tools = [buscar_propiedad, listar_zonas]
agent = create_react_agent(llm, tools, prompt)
history = ChatMessageHistory()
with_history = RunnableWithMessageHistory(agent, lambda session_id: history)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- INTERFAZ DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Escribe tu consulta sobre propiedades...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Buscando información..."):
            response = executor.invoke({"input": user_input})
            st.markdown(response["output"])
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})

# --- HISTORIAL ---
if st.session_state.messages:
    with st.expander("📝 Historial de conversación"):
        for msg in st.session_state.messages:
            role = "👤" if msg["role"] == "user" else "🤖"
            st.write(f"**{role}**: {msg['content']}")
