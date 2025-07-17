from pydantic import BaseModel, Field
from datetime import datetime


class CropYieldPayload(BaseModel):
    model: str
    country: str = Field(min_length=5, max_length=100)
    year: int = Field(ge=1900, le=datetime.now().year)
    element: str = Field(min_length=5, max_length=100)
    average_rain_fall_mm_per_year: int = Field(ge=0)
    pesticide_tonnes: float = Field(ge=0)
    avg_temp_in_celsius: float