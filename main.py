"""RBAC web entry point for the RAG Assistant."""
import json, os, threading
import gradio as gr
from src.auth import AuthService

auth = AuthService(); _rag = None; _rag_lock = threading.Lock()
def get_rag():
    global _rag
    with _rag_lock:
        if _rag is None:
            from src.rag_orchestrator import RAGCoordinator
            _rag = RAGCoordinator(session_id="web-session", project_id=os.getenv("GOOGLE_CLOUD_PROJECT", "august-balancer-460307-q0"), index_path="vector_database", metadata_json="vector_database/doc_metadata.json")
    return _rag
def login(username,password):
    token=auth.authenticate(username,password); user=auth.user_for_token(token)
    if not user:return "Invalid credentials",gr.update(visible=True),gr.update(visible=False),"",None
    return f"Signed in as **{user.username}** ({user.role})",gr.update(visible=False),gr.update(visible=True),token,user.role
def logout():return "Signed out",gr.update(visible=True),gr.update(visible=False),"",None
def create_account(token,username,password,role):
    try:return auth.create_user(auth.user_for_token(token),username,password,role)
    except Exception as exc:return f"Error: {exc}"
def load_metadata(token):
    if not auth.can(auth.user_for_token(token),"metadata"):return "Not authorized"
    with open("vector_database/doc_metadata.json",encoding="utf-8") as fh:return json.dumps(json.load(fh),indent=2,ensure_ascii=False)
def save_metadata(token,text):
    if not auth.can(auth.user_for_token(token),"metadata"):return "Not authorized"
    try:
        data=json.loads(text)
        with open("vector_database/doc_metadata.json","w",encoding="utf-8") as fh:json.dump(data,fh,indent=2,ensure_ascii=False)
        return "Metadata saved. Rebuild the vector database to apply changes."
    except Exception as exc:return f"Invalid JSON: {exc}"
def rebuild(token):
    if not auth.can(auth.user_for_token(token),"vector"):return "Not authorized"
    try:
        from src.build_vector_db import VectorDBBuilder
        VectorDBBuilder(os.getenv("GOOGLE_CLOUD_PROJECT","august-balancer-460307-q0")).create_database("vector_database/doc_metadata.json","vector_database")
        return "Vector database updated successfully."
    except Exception as exc:return f"Vector update failed: {exc}"
def chat(token,message,history):
    if not auth.can(auth.user_for_token(token),"chat"):return history,"Not authorized"
    if not message.strip():return history,""
    return history+[{"role":"user","content":message},{"role":"assistant","content":get_rag().ask(message)}],""

with gr.Blocks(title="RAG Assistant") as demo:
    token=gr.State(""); role=gr.State(None)
    with gr.Column(visible=True) as login_view:
        gr.Markdown("# RAG Assistant\nSign in to access your role-scoped workspace.")
        username=gr.Textbox(label="Username",value="master"); password=gr.Textbox(label="Password",type="password")
        login_btn=gr.Button("Sign in",variant="primary"); login_status=gr.Markdown()
    with gr.Column(visible=False) as app_view:
        gr.Markdown("# RAG Assistant — Control plane"); logout_btn=gr.Button("Sign out",size="sm")
        with gr.Tabs():
            with gr.Tab("Accounts"):
                gr.Markdown("Master creates Managers and General users; Managers create General users.")
                account_role=gr.Dropdown(["manager","general"],label="Role",value="general"); account_user=gr.Textbox(label="Username"); account_pass=gr.Textbox(label="Temporary password",type="password")
                account_btn=gr.Button("Create account"); account_status=gr.Markdown()
            with gr.Tab("Knowledge base"):
                metadata=gr.Code(label="Document metadata (JSON)",language="json",lines=18)
                with gr.Row():load_btn=gr.Button("Load metadata");save_btn=gr.Button("Save metadata")
                index_btn=gr.Button("Update vector database",variant="primary");kb_status=gr.Markdown()
            with gr.Tab("Chat"):
                chatbot=gr.Chatbot(label="RAG conversation",height=500);question=gr.Textbox(label="Question");send=gr.Button("Send",variant="primary");chat_status=gr.Markdown()
    login_btn.click(login,[username,password],[login_status,login_view,app_view,token,role]);logout_btn.click(logout,None,[login_status,login_view,app_view,token,role])
    account_btn.click(create_account,[token,account_user,account_pass,account_role],account_status);load_btn.click(load_metadata,token,metadata);save_btn.click(save_metadata,[token,metadata],kb_status);index_btn.click(rebuild,token,kb_status)
    send.click(chat,[token,question,chatbot],[chatbot,chat_status]);question.submit(chat,[token,question,chatbot],[chatbot,chat_status])
if __name__=="__main__":demo.launch()
