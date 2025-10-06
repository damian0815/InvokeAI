import PIL.Image

from invokeai.app.invocations.baseinvocation import invocation, BaseInvocation
from invokeai.app.invocations.fields import WithBoard, TensorField, InputField, ImageField
from invokeai.app.invocations.primitives import ImageCollectionOutput
from invokeai.app.services.images.images_common import ImageDTO
from invokeai.app.services.shared.invocation_context import InvocationContext


@invocation(
    "save_attention_maps",
    title="Save Attention Maps",
    tags=["latents", "attention", "txt2img", "t2i"],
    category="latents",
    version="0.0.1",
)
class SaveAttentionMapsInvocation(BaseInvocation, WithBoard):
    """ Saves attention maps as images. """
    attention_maps: list[TensorField] = InputField(description="The attention maps to save")

    def invoke(self, context: InvocationContext) -> ImageCollectionOutput:
        image_names = []
        for tensor_field in self.attention_maps:
            attention_map = context.tensors.load(tensor_field.tensor_name)
            attention_map = attention_map.mul(0xff).byte()
            image_dto: ImageDTO = context.images.save(image=PIL.Image.fromarray(attention_map.numpy(), mode='L'))
            image_names.append(image_dto.image_name)

        return ImageCollectionOutput(collection=[ImageField(image_name=name) for name in image_names])
