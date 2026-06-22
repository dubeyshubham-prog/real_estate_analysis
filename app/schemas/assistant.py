from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class AssistantInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=3, max_length=1_000)
