"""main_vector_db.py – Build / rebuild the FAISS vector database."""
from src.build_vector_db import VectorDBBuilder

if __name__ == "__main__":
    PROJECT_ID    = "august-balancer-460307-q0"
    JSON_METADATA = "vector_database/doc_metadata.json"
    DB_FOLDER     = "vector_database"

    builder = VectorDBBuilder(project_id=PROJECT_ID)
    builder.create_database(JSON_METADATA, DB_FOLDER)
