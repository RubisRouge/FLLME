from pydantic import BaseModel
from enum import StrEnum


class Ratio(StrEnum):
    _1_1 = "1:1"
    _2_3 = "2:3"
    _3_2 = "3:2"
    _3_4 = "3:4"
    _4_3 = "4:3"
    _9_16 = "9:16"
    _16_9 = "16:9"
    _21_9 = "21:9"


class Resolution(StrEnum):
    _1K = "1K"
    _2K = "2K"
    _4K = "4K"


class PersonGeneration(StrEnum):
    ALL = "ALLOW_ALL"
    ADULT = "ALLOW_ADULT"
    NONE = "ALLOW_NONE"


class MimeType(StrEnum):
    PNG = "image/png"
    JPEG = "image/jpeg"


class ImageConfig(BaseModel):
    ratio: Ratio = Ratio._1_1
    resolution: Resolution = Resolution._1K
    person_generation: PersonGeneration = PersonGeneration.ALL
    mime_type: MimeType = MimeType.PNG
