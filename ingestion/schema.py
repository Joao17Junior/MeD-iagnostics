import uuid
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from PIL import Image

class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any]
    image: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
