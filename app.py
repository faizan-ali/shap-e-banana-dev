from potassium import Potassium, Request, Response
import base64

import torch

from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, decode_latent_mesh

app = Potassium("my_app")


# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xm = load_model('transmitter', device=device)
    model = load_model('text300M', device=device)
    diffusion = diffusion_from_config(load_config('diffusion'))

    context = {
        "model": model,
        "diffusion": diffusion,
        "device": device,
        "xm": xm
    }

    return context


# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    diffusion = context.get("diffusion")
    device = context.get("device")
    xm = context.get("xm")

    print('Generating 3D model for prompt: ', prompt)

    batch_size = 1
    guidance_scale = 15.0
    render_mode = 'nerf'
    size = 64

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(texts=[prompt] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1E-3,
        sigma_max=160,
        s_churn=0,
    )

    cameras = create_pan_cameras(size, device)
    for i, latent in enumerate(latents):
        images = decode_latent_images(xm, latent, cameras, rendering_mode=render_mode)

    filename = f'tmp_mesh.obj'

    t = decode_latent_mesh(xm, latents[0]).tri_mesh()
    with open(filename, 'w') as f:
        t.write_obj(f)

    print('3D asset generated')

    output = base64.b64encode(f)

    return Response(
        json={"output": output},
        status=200
    )


if __name__ == "__main__":
    app.serve()
