from ingestion.parsers.dicom_parser import DICOM_Parser

# Substitui pelo caminho exato do teu ficheiro
DICOM_PATH = "data/0002.DCM"

parser = DICOM_Parser()
chunk = parser.parse(DICOM_PATH)

print("=== TESTE COM DICOM REAL ===")
print(f"ID Gerado: {chunk.id}")
print(f"Conteúdo (Resumo RAG):\n{chunk.content}")
print(f"\nTotal de Tags Extraídas nos Metadados: {len(chunk.metadata)}")

if chunk.image:
    print(f"\nImagem Normalizada:")
    print(f"• Dimensões: {chunk.image.size} (Largura x Altura)")
    print(f"• Modo de Cor: {chunk.image.mode}")
