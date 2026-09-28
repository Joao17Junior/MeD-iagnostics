from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import Any, Dict, List, Optional
import torch
from PIL import Image

class Med_VLM:
    def __init__ (
            self,
            model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
            device: Optional[str] = None
            ):
        self.model_path = model_path
        self.device = (device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[Qwen2_5_VLForConditionalGeneration] = None

    def load_model (self) -> None:

        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                device_map = "auto" if self.device == "cuda" else None
                )

        if self.device == "cpu":
            self.model.to(self.device)


    def generate_response (
            self,
            query: str,
            retrieved_chunks: List[Dict[str, Any]],
            image: Optional[Image.Image] = None,
            max_new_tokens: int = 512
            ) -> str:

        if self.model is None or self.processor is None:
            self.load_model()

        # here we can call the hybrid query in the retrieved chunks for the data thats similar and we can keep it in stack (to implement)
        # for now we shall use the retrieved_chunks as context and not as thought in the line above
        # TODO: build context [], create the prompt, prepare input payload, process input and gen output
