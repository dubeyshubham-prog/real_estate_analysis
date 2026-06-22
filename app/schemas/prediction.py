from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator


class PredictionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    property_type: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    bedRoom: float = Field(gt=0, le=20)
    bathroom: float = Field(gt=0, le=20)
    balcony: float = Field(ge=0, le=10)
    agePossession: str = Field(min_length=1)
    floor_num: float = Field(ge=-2, le=100)
    total_floors: float = Field(ge=0, le=100)
    built_up_area: float = Field(gt=100, le=100_000)
    servant_room: int = Field(ge=0, le=1)
    store_room: int = Field(ge=0, le=1)
    furnishing_type: str = Field(min_length=1)
    luxury_category: str = Field(min_length=1)
    floor_category: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_floor_relationship(self) -> "PredictionInput":
        if self.total_floors and self.floor_num > self.total_floors:
            raise ValueError(
                "floor_num cannot be greater than total_floors"
            )
        return self

    def to_model_input(self) -> dict[str, object]:
        values = self.model_dump()
        values["servant room"] = values.pop("servant_room")
        values["store room"] = values.pop("store_room")
        return values
