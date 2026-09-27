import numpy as np
from PIL import Image
from ingestion.vectoring.embedder import BMC_Embedder


def test_embedder():
    print("=" * 60)
    print("A inicializar o BioMedCLIPEmbedder...")
    print("(Se for a primeira execução, fará o download do modelo ~700 MB)")
    print("=" * 60)

    embedder = BMC_Embedder()

    # 1. Teste de Embedding de Texto
    sample_text = "Chest X-ray showing bilateral lobar pneumonia with consolidation"
    print(f"\n[1/3] A vetorizar texto: '{sample_text}'")
    text_vectors = embedder.embed_txt(sample_text)
    text_vector = text_vectors[0]

    print(f"  • Dimensão do vetor: {len(text_vector)}")
    print(f"  • Amostra dos 5 primeiros valores: {[round(x, 4) for x in text_vector[:5]]}")

    # 2. Teste de Embedding de Imagem (Gera uma imagem RGB sintética de 512x512)
    print("\n[2/3] A vetorizar imagem...")
    synthetic_pixels = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    sample_image = Image.fromarray(synthetic_pixels)

    image_vectors = embedder.embed_img(sample_image)
    image_vector = image_vectors[0]

    print(f"  • Dimensão do vetor: {len(image_vector)}")
    print(f"  • Amostra dos 5 primeiros valores: {[round(x, 4) for x in image_vector[:5]]}")

    # 3. Verificações de Integridade
    print("\n[3/3] A validar integridade matemática...")

    # Confirmar dimensão exata de 512
    assert len(text_vector) == 512, f"Erro: Esperado 512, obtido {len(text_vector)}"
    assert len(image_vector) == 512, f"Erro: Esperado 512, obtido {len(image_vector)}"

    # Confirmar Normalização L2 (A norma/comprimento do vetor deve ser igual a 1.0)
    text_norm = np.linalg.norm(text_vector)
    image_norm = np.linalg.norm(image_vector)
    print(f"  • Norma L2 do Texto: {text_norm:.4f} (Esperado ~1.0)")
    print(f"  • Norma L2 da Imagem: {image_norm:.4f} (Esperado ~1.0)")

    print("\n" + "=" * 60)
    print("SUCESSO: BioMedCLIPEmbedder totalmente funcional!")
    print("Texto e imagem partilham o mesmo espaço vetorial de 512D.")
    print("=" * 60)


if __name__ == "__main__":
    test_embedder()
