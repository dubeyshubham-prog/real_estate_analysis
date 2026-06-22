from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class RecommendationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    property_name: str = Field(min_length=1, max_length=200)
    top_n: int = Field(default=5, ge=1, le=20)
