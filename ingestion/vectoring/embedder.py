from typing import List, Union
import torch
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

class BMC_Embedder:
    def __init__(self, model_name: str="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"):
        # if has cuda (GPU) uses cuda, else uses CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
        # loads model
        self.model, self.preprocess = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer(model_name)

        # sends model to device mem
        self.model.to(self.device)
        self.model.eval()

    def embed_txt(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        context_length = 256
        tokens = self.tokenizer(texts, context_length=context_length).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(tokens)

            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features.cpu().numpy().tolist()

    def embed_img(self, images: Union[Image.Image, List[Image.Image]]) -> List[List[float]]:
        if isinstance(images, Image.Image):
            images = [images]

        image_tensors = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)

            image_features /= image_features.norm(dim=1, keepdim=True)

        return image_features.cpu().numpy().tolist()
