from comfy_api.latest import ComfyExtension

from .nodes import SingleEncoderEnhancer

WEB_DIRECTORY = "./web"


class SingleEncoderEnhancerExtension(ComfyExtension):
    async def get_node_list(self):
        return [SingleEncoderEnhancer]


async def comfy_entrypoint():
    return SingleEncoderEnhancerExtension()
