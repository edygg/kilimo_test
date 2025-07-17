from pydantic import BaseModel

from kilimo_test.api.payloads import CropYieldPayload


class CropYieldPredictedResponse(CropYieldPayload):
    hg_ha_yield: float