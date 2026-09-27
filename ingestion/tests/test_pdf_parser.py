from src.ingestion.parsers.pdf_parser import PDF_Parser

parser = PDF_Parser(chunk_size=300, chunk_overlap=30)
# Passa o caminho de qualquer PDF que tenhas no teu computador
chunks = parser.parse("data/raw/exemplo_artigo.pdf", doc_type="literature")

print(f"Total de chunks gerados: {len(chunks)}")
if chunks:
    print("\n--- Exemplo do Primeiro Chunk ---")
    print(f"ID: {chunks[0].id}")
    print(f"Conteúdo: {chunks[0].content}")
    print(f"Metadados: {chunks[0].metadata}")
