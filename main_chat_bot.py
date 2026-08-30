"""main_chat_bot.py – CLI chat loop."""
from src.rag_orchestrator import RAGCoordinator

if __name__ == "__main__":
    rag = RAGCoordinator(
        session_id="cli-default",
        project_id="august-balancer-460307-q0",
        index_path="vector_database",
        metadata_json="vector_database/doc_metadata.json",
    )

    while True:
        user_in = input("Domanda (o 'exit'): ").strip()
        if user_in.lower() == "exit":
            print("Arrivederci!")
            break
        print("\n" + "─" * 60)
        print(rag.ask(user_in))
        print("─" * 60 + "\n")
