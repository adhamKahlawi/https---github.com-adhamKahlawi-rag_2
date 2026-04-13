"""
main.py  –  Gradio front-end for the Gemini RAG Assistant.
"""

import gradio as gr
from src.chat_llm import GeminiRAGSystem

# ── Initialise once ────────────────────────────────────────────────────────────
rag = GeminiRAGSystem(
    project_id="august-balancer-460307-q0",
    index_path="vector_database",
    metadata_json="vector_database/doc_metadata.json",
)


# ── Chat handler ───────────────────────────────────────────────────────────────
def chat_engine(user_input: str, history: list):
    if not user_input.strip():
        return "", history

    answer = rag.ask(user_input)

    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": answer})
    return "", history


# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="RAG Assistant") as demo:
    gr.Markdown(
        """
        ### Benvenuto nel RAG Assistant

        Il sistema rileva automaticamente se vuoi una **risposta puntuale** (QA)
        o un **riassunto** di un documento (basta includere il titolo nella domanda).
        """
    )

    chatbot = gr.Chatbot(label="Conversazione", height=500) #, type="messages"

    msg = gr.Textbox(
        label="La tua domanda",
        placeholder="es. 'come si configura X?' oppure 'Riassumi INTRODUZIONE A FAIR'",
        lines=2,
    )

    with gr.Row():
        submit_btn = gr.Button("Invia", variant="primary")
        clear_btn  = gr.Button("Nuova chat")

    msg.submit(chat_engine, [msg, chatbot], [msg, chatbot])
    submit_btn.click(chat_engine, [msg, chatbot], [msg, chatbot])
    clear_btn.click(lambda: ([], []), None, [chatbot], queue=False)

if __name__ == "__main__":
    demo.launch()
