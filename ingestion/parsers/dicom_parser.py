import pydicom
import uuid
import numpy as np
from typing import Optional, Any, Dict
from PIL import Image
from ingestion.schema import DocumentChunk

class DICOM_Parser:
    def parse(self, file_path: str) -> DocumentChunk:
        ds = pydicom.dcmread(file_path)

        extracted_metadata: Dict[str, Any] = {
                "source": file_path,
                "type": "dicom_scan",
                }
        summary_lines = []

        # run through the DICOM file
        for elem in ds:
            # leave out of the chunk all img processing data (hopefully gemini cooked)
            if elem.VR in ["OB", "OW", "OF", "UT", "UN"] or elem.tag == (0x7FE0, 0x0010):
                continue

            key = elem.keyword or f"Tag_{elem.tag.group:04X}_{elem.tag.element:04X}"
            val = elem.value
    
            # pass the elements as primitive type
            if isinstance(val, (pydicom.multival.MultiValue, list)):
                val = ", ".join(str(v) for v in val)
            elif not isinstance(val, (int, float, str, bool)):
                val = str(val)


            extracted_metadata[key] = val
            summary_lines.append(f"{key}: {val}")

        # text resume of all tags found
        metadata_summary = "DICOM Scan Metadata:\n" + "\n".join(summary_lines)

        # process image for BioMedClip
        pil_image: Optional[Image.Image] = None
        if hasattr(ds, "pixel_array"):
            pixel_array = ds.pixel_array.astype(float)
            pixel_array = np.squeeze(pixel_array)

            if pixel_array.ndim == 3 and pixel_array.shape[-1] not in (3, 4):
                middle_idx = pixel_array.shape[0] // 2
                pixel_array = pixel_array[middle_idx]

            if pixel_array.ndim == 1:
                pixel_array = np.expand_dims(pixel_array, axis=0)

            min_val = pixel_array.min()
            max_val = pixel_array.max()

            if max_val > min_val:
                rescaled = ((pixel_array - min_val) / (max_val - min_val)) * 255.0
            else:
                rescaled = np.zeros_like(pixel_array)

            rescaled_uint8 = np.uint8(rescaled)

            if rescaled_uint8.ndim == 2:
                pil_image = Image.fromarray(rescaled_uint8).convert("RGB")
            elif rescaled_uint8.ndim == 3 and rescaled_uint8.shape[-1] in (3, 4):
                pil_image = Image.fromarray(rescaled_uint8).convert("RGB")


        return DocumentChunk(
                id       = str(uuid.uuid4()),
                content  = metadata_summary,
                metadata = extracted_metadata,
                image    = pil_image
                )
