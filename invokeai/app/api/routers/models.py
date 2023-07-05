# Copyright (c) 2023 Kyle Schouviller (https://github.com/kyle0654) and 2023 Kent Keirsey (https://github.com/hipsterusername)

from typing import Literal, Optional, Union

from fastapi import Query, Body, Path
from fastapi.routing import APIRouter, HTTPException
from pydantic import BaseModel, Field, parse_obj_as
from ..dependencies import ApiDependencies
from invokeai.backend import BaseModelType, ModelType
from invokeai.backend.model_management import AddModelResult
from invokeai.backend.model_management.models import MODEL_CONFIGS, OPENAPI_MODEL_CONFIGS, SchedulerPredictionType

models_router = APIRouter(prefix="/v1/models", tags=["models"])

class CreateModelResponse(BaseModel):
    model_name: str = Field(description="The name of the new model")
    info: Union[tuple(MODEL_CONFIGS)] = Field(description="The model info")
    status: str = Field(description="The status of the API response")

class ImportModelResponse(BaseModel):
    name: str = Field(description="The name of the imported model")
    info: AddModelResult = Field(description="The model info")
    status: str = Field(description="The status of the API response")

class ConvertModelResponse(BaseModel):
    name: str = Field(description="The name of the imported model")
    info: AddModelResult = Field(description="The model info")
    status: str = Field(description="The status of the API response")
    
class ModelsList(BaseModel):
    models: list[Union[tuple(OPENAPI_MODEL_CONFIGS)]]

@models_router.get(
    "/{base_model}/{model_type}",
    operation_id="list_models",
    responses={200: {"model": ModelsList }},
)
async def list_models(
    base_model: Optional[BaseModelType] = Path(
        default=None, description="Base model"
    ),
    model_type: Optional[ModelType] = Path(
        default=None, description="The type of model to get"
    ),
) -> ModelsList:
    """Gets a list of models"""
    models_raw = ApiDependencies.invoker.services.model_manager.list_models(base_model, model_type)
    models = parse_obj_as(ModelsList, { "models": models_raw })
    return models

@models_router.post(
    "/{base_model}/{model_type}/{model_name}",
    operation_id="update_model",
    responses={200: {"status": "success"}},
)
async def update_model(
        base_model: BaseModelType = Path(default='sd-1', description="Base model"),
        model_type: ModelType = Path(default='main', description="The type of model"),
        model_name: str = Path(default=None, description="model name"),
        info: Union[tuple(MODEL_CONFIGS)]  = Body(description="Model configuration"),
) -> CreateModelResponse:
    """ Add Model """
    ApiDependencies.invoker.services.model_manager.add_model(
        model_name=model_name,
        base_model=base_model,
        model_type=model_type,
        model_attributes=info.dict(),
        clobber=True,
    )
    model_response = CreateModelResponse(
        model_name = model_name,
        info = info,
        status="success")

    return model_response

@models_router.post(
    "/import",
    operation_id="import_model",
    responses= {
        201: {"description" : "The model imported successfully"},
        404: {"description" : "The model could not be found"},
        409: {"description" : "There is already a model corresponding to this path or repo_id"},
    },
    status_code=201,
    response_model=ImportModelResponse
)
async def import_model(
        name: str = Body(description="A model path, repo_id or URL to import"),
        prediction_type: Optional[Literal['v_prediction','epsilon','sample']] = \
                Body(description='Prediction type for SDv2 checkpoint files', default="v_prediction"),
) -> ImportModelResponse:
    """ Add a model using its local path, repo_id, or remote URL """
    
    items_to_import = {name}
    prediction_types = { x.value: x for x in SchedulerPredictionType }
    logger = ApiDependencies.invoker.services.logger

    try:
        installed_models = ApiDependencies.invoker.services.model_manager.heuristic_import(
            items_to_import = items_to_import,
            prediction_type_helper = lambda x: prediction_types.get(prediction_type)
        )
        if info := installed_models.get(name):
            logger.info(f'Successfully imported {name}, got {info}')
            return ImportModelResponse(
                name = name,
                info = info,
                status = "success",
        )
    except KeyError as e:
        logger.error(str(e))
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(status_code=409, detail=str(e))
        

@models_router.delete(
    "/{base_model}/{model_type}/{model_name}",
    operation_id="del_model",
    responses={
        204: {
        "description": "Model deleted successfully"
        }, 
        404: {
        "description": "Model not found"
        }
    },
)
async def delete_model(
        base_model: BaseModelType = Path(default='sd-1', description="Base model"),
        model_type: ModelType = Path(default='main', description="The type of model"),
        model_name: str = Path(default=None, description="model name"),
) -> None:
    """Delete Model"""
    logger = ApiDependencies.invoker.services.logger
    
    try:
        ApiDependencies.invoker.services.model_manager.del_model(model_name,
                                                                 base_model = base_model,
                                                                 model_type = model_type
                                                                 )
        logger.info(f"Deleted model: {model_name}")
        raise HTTPException(status_code=204, detail=f"Model '{model_name}' deleted successfully")
    except KeyError:
        logger.error(f"Model not found: {model_name}")
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    else:
        logger.info(f"Model deleted: {model_name}")
        raise HTTPException(status_code=204, detail=f"Model '{model_name}' deleted successfully")
    

            # @socketio.on("convertToDiffusers")
        # def convert_to_diffusers(model_to_convert: dict):
        #     try:
        #         if model_info := self.generate.model_manager.model_info(
        #             model_name=model_to_convert["model_name"]
        #         ):
        #             if "weights" in model_info:
        #                 ckpt_path = Path(model_info["weights"])
        #                 original_config_file = Path(model_info["config"])
        #                 model_name = model_to_convert["model_name"]
        #                 model_description = model_info["description"]
        #             else:
        #                 self.socketio.emit(
        #                     "error", {"message": "Model is not a valid checkpoint file"}
        #                 )
        #         else:
        #             self.socketio.emit(
        #                 "error", {"message": "Could not retrieve model info."}
        #             )

        #         if not ckpt_path.is_absolute():
        #             ckpt_path = Path(Globals.root, ckpt_path)

        #         if original_config_file and not original_config_file.is_absolute():
        #             original_config_file = Path(Globals.root, original_config_file)

        #         diffusers_path = Path(
        #             ckpt_path.parent.absolute(), f"{model_name}_diffusers"
        #         )

        #         if model_to_convert["save_location"] == "root":
        #             diffusers_path = Path(
        #                 global_converted_ckpts_dir(), f"{model_name}_diffusers"
        #             )

        #         if (
        #             model_to_convert["save_location"] == "custom"
        #             and model_to_convert["custom_location"] is not None
        #         ):
        #             diffusers_path = Path(
        #                 model_to_convert["custom_location"], f"{model_name}_diffusers"
        #             )

        #         if diffusers_path.exists():
        #             shutil.rmtree(diffusers_path)

        #         self.generate.model_manager.convert_and_import(
        #             ckpt_path,
        #             diffusers_path,
        #             model_name=model_name,
        #             model_description=model_description,
        #             vae=None,
        #             original_config_file=original_config_file,
        #             commit_to_conf=opt.conf,
        #         )

        #         new_model_list = self.generate.model_manager.list_models()
        #         socketio.emit(
        #             "modelConverted",
        #             {
        #                 "new_model_name": model_name,
        #                 "model_list": new_model_list,
        #                 "update": True,
        #             },
        #         )
        #         print(f">> Model Converted: {model_name}")
        #     except Exception as e:
        #         self.handle_exceptions(e)

        # @socketio.on("mergeDiffusersModels")
        # def merge_diffusers_models(model_merge_info: dict):
        #     try:
        #         models_to_merge = model_merge_info["models_to_merge"]
        #         model_ids_or_paths = [
        #             self.generate.model_manager.model_name_or_path(x)
        #             for x in models_to_merge
        #         ]
        #         merged_pipe = merge_diffusion_models(
        #             model_ids_or_paths,
        #             model_merge_info["alpha"],
        #             model_merge_info["interp"],
        #             model_merge_info["force"],
        #         )

        #         dump_path = global_models_dir() / "merged_models"
        #         if model_merge_info["model_merge_save_path"] is not None:
        #             dump_path = Path(model_merge_info["model_merge_save_path"])

        #         os.makedirs(dump_path, exist_ok=True)
        #         dump_path = dump_path / model_merge_info["merged_model_name"]
        #         merged_pipe.save_pretrained(dump_path, safe_serialization=1)

        #         merged_model_config = dict(
        #             model_name=model_merge_info["merged_model_name"],
        #             description=f'Merge of models {", ".join(models_to_merge)}',
        #             commit_to_conf=opt.conf,
        #         )

        #         if vae := self.generate.model_manager.config[models_to_merge[0]].get(
        #             "vae", None
        #         ):
        #             print(f">> Using configured VAE assigned to {models_to_merge[0]}")
        #             merged_model_config.update(vae=vae)

        #         self.generate.model_manager.import_diffuser_model(
        #             dump_path, **merged_model_config
        #         )
        #         new_model_list = self.generate.model_manager.list_models()

        #         socketio.emit(
        #             "modelsMerged",
        #             {
        #                 "merged_models": models_to_merge,
        #                 "merged_model_name": model_merge_info["merged_model_name"],
        #                 "model_list": new_model_list,
        #                 "update": True,
        #             },
        #         )
        #         print(f">> Models Merged: {models_to_merge}")
        #         print(f">> New Model Added: {model_merge_info['merged_model_name']}")
        #     except Exception as e:
