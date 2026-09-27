import os
import shutil
from PIL import Image
import numpy as np

from ingestion.schema import DocumentChunk
from ingestion.vectoring.embedder import BMC_Embedder
from ingestion.storer.chroma_storer import Chroma_Storer


def test_chroma_storer():
    test_db_path = "./test_chroma_db"

    # Limpar base de dados de teste anterior, se existir
    if os.path.exists(test_db_path):
        shutil.rmtree(test_db_path)

    print("=" * 60)
    print("A inicializar Embedder e ChromaStorer...")
    embedder = BMC_Embedder()
    storer = Chroma_Storer(db_path=test_db_path, collection_name="test_collection")

    # 1. Criar Chunks de Teste (Texto de Literatura + Metadados de Exame)
    chunk1 = DocumentChunk(
        content="Bilateral pulmonary opacities observed in chest radiography, suggestive of acute viral pneumonia.",
        metadata={"source": "pubmed_article_102.pdf", "type": "literature"},
    )

    sample_img = Image.fromarray(np.uint8(np.random.rand(224, 224) * 255)).convert("RGB")
    chunk2 = DocumentChunk(
        content="DICOM Scan Metadata: Modality: CT | Body Part: CHEST | Patient Sex: M",
        metadata={"source": "patient_882.dcm", "modality": "CT", "type": "dicom_scan"},
        image=sample_img,
    )

    # 2. Gerar Embeddings Multimodais
    print("\n[1/3] A gerar vetores multimodais (Texto e Imagem)...")
    embedding1 = embedder.embed_txt(chunk1.content)[0]
    embedding2 = embedder.embed_img(chunk2.image)[0]

    # 3. Guardar no ChromaDB
    print("[2/3] A guardar chunks na coleção ChromaDB...")
    storer.add_chunks(
        chunks=[chunk1, chunk2],
        embeddings=[embedding1, embedding2],
    )

    total_count = storer.count()
    print(f"  • Total de documentos guardados: {total_count}")
    assert total_count == 2

    # 4. Fazer uma Consulta de Teste por Texto
    print("\n[3/3] A testar consulta por similaridade semântica...")
    query_text = "chest x-ray viral infection"
    query_vector = embedder.embed_txt(query_text)[0]

    results = storer.query_similar(query_embedding=query_vector, n_results=1)

    matched_doc = results["documents"][0][0]
    matched_id = results["ids"][0][0]
    distance = results["distances"][0][0]

    print(f"  • Consulta: '{query_text}'")
    print(f"  • Resultado mais próximo (ID: {matched_id}):")
    print(f"    '{matched_doc}'")
    print(f"  • Distância de Cosseno: {distance:.4f}")

    # Limpeza
    # shutil.rmtree(test_db_path)
    print("\n" + "=" * 60)
    print("SUCESSO: ChromaStorer a funcionar perfeitamente com o BioMedCLIP!")
    print("=" * 60)


if __name__ == "__main__":
    test_chroma_storer()
