from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from PIL import Image

class DocumentChunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    image: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
