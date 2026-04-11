"""main_doc_metadata.py – Generate / refresh document metadata."""
from src.build_doc_metadata import MetadataGenerator

if __name__ == "__main__":
    PROJECT_ID        = "august-balancer-460307-q0"
    INPUT_DATA        = "input_data"
    JSON_METADATA     = "vector_database/doc_metadata.json"

    generator = MetadataGenerator(project_id=PROJECT_ID)
    generator.generate_metadata(INPUT_DATA, JSON_METADATA)
