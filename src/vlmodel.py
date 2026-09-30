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

        # context
        context_parts = []
        for i, item in enumerate(retrieved_chunks, start=1):
            chunk = item["chunk"]
            context_parts.append(f"--- Doc {i} ---\n- content: {chunk.content}\n- metadata: {chunk.metadata}")
        
        context_str = "\n\n".join(context_parts)

        # prompt
        prompt_text = (
                "You are a medical assistent specialized in diagnostics.\n"
                "Answer the question posted by the user in a clear and objective way, "
                "backing yourself up EXCLUSIVELY with the clinical context given "
                "(and in the image sent, if applicable).\n"
                "If the information is not available in the context, declare explicitly that "
                "you have not found it, so you shall not give an answer!\n\n"
                f"### Clinical Context Retrieved:\n{context_str}\n\n"
                f"### User Question:\n{query}"
                )


        # content payload
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        # process input
        text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
                )

        inputs = self.processor(
                text=[text_prompt],
                images=[image] if image is not None else None,
                padding=True,
                return_tensors="pt"
                ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                    )

        # remove prompt tokens
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

        output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens = True,
                clean_up_tokenization_spaces = False
                )

        print(output_text)

        return output_text[0]


    
