from fastapi import FastAPI, HTTPException, status, Query, Body

from kilimo_test.api.payloads import CropYieldPayload
from kilimo_test.api.responses import CropYieldPredictedResponse
from kilimo_test.ml.models import get_model, get_predict_obj, get_label_encoder
from kilimo_test.ml.training import get_models_local_storage
from kilimo_test.logging import logger

app = FastAPI(
    title="Kilimo ML Test by Edilson Gonzalez",
)


@app.get("/health-check")
async def health_check():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
        payload: CropYieldPayload,
):
    model = payload.model
    try:
        model_obj = get_model(
            model_name=model,
            storage_dir=get_models_local_storage(),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    label_encoder_obj = get_label_encoder(get_models_local_storage())
    payload_to_predict = payload.model_dump(exclude={"model"})

    try:
        payload_to_predict["country"] = label_encoder_obj.transform([payload_to_predict["country"]])[0]
        payload_to_predict["element"] = label_encoder_obj.transform([payload_to_predict["element"]])[0]
    except Exception as e:
        logger.error(e)
        logger.info(list(label_encoder_obj.classes_))

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid country or element, does not fit in the model",
        )

    predict_obj = get_predict_obj(
        payload_to_predict
    )

    predicted = model_obj.predict(
        predict_obj
    )

    return CropYieldPredictedResponse.model_validate({
        **payload.model_dump(),
        "hg_ha_yield": predicted[0] if len(predicted) == 1 else -1.0
    })