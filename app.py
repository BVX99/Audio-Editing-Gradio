import gradio as gr
import random
import torch
import torchaudio
from torch import inference_mode
from tempfile import NamedTemporaryFile
import numpy as np
from models import load_model
import utils
from inversion_utils import inversion_forward_process, inversion_reverse_process


# current_loaded_model = "cvssp/audioldm2-music"
# # current_loaded_model = "cvssp/audioldm2-music"

# ldm_stable = load_model(current_loaded_model, device, 200)  # deafult model
LDM2 = "cvssp/audioldm2"
MUSIC = "cvssp/audioldm2-music"
LDM2_LARGE = "cvssp/audioldm2-large"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ldm2 = load_model(model_id=LDM2, device=device)
ldm2_large = load_model(model_id=LDM2_LARGE, device=device)
ldm2_music = load_model(model_id=MUSIC, device=device)


def randomize_seed_fn(seed, randomize_seed):
    if randomize_seed:
        seed = random.randint(0, np.iinfo(np.int32).max)
    torch.manual_seed(seed)
    return seed


def invert(ldm_stable, x0, prompt_src, num_diffusion_steps, cfg_scale_src):  # , ldm_stable):
    ldm_stable.model.scheduler.set_timesteps(num_diffusion_steps, device=device)

    with inference_mode():
        w0 = ldm_stable.vae_encode(x0)

    # find Zs and wts - forward process
    _, zs, wts = inversion_forward_process(ldm_stable, w0, etas=1,
                                           prompts=[prompt_src],
                                           cfg_scales=[cfg_scale_src],
                                           prog_bar=True,
                                           num_inference_steps=num_diffusion_steps,
                                           numerical_fix=True)
    return zs, wts


def sample(ldm_stable, zs, wts, steps, prompt_tar, tstart, cfg_scale_tar):  # , ldm_stable):
    # reverse process (via Zs and wT)
    tstart = torch.tensor(tstart, dtype=torch.int)
    skip = steps - tstart
    w0, _ = inversion_reverse_process(ldm_stable, xT=wts, skips=steps - skip,
                                      etas=1., prompts=[prompt_tar],
                                      neg_prompts=[""], cfg_scales=[cfg_scale_tar],
                                      prog_bar=True,
                                      zs=zs[:int(steps - skip)])

    # vae decode image
    with inference_mode():
        x0_dec = ldm_stable.vae_decode(w0)
    if x0_dec.dim() < 4:
        x0_dec = x0_dec[None, :, :, :]

    with torch.no_grad():
        audio = ldm_stable.decode_to_mel(x0_dec)

    f = NamedTemporaryFile("wb", suffix=".wav", delete=False)
    torchaudio.save(f.name, audio, sample_rate=16000)

    return f.name



def edit(input_audio,
         model_id: str,
         do_inversion: bool,
         wts: gr.State, zs: gr.State, saved_inv_model: str,
         source_prompt="",
         target_prompt="",
         steps=200,
         cfg_scale_src=3.5,
         cfg_scale_tar=12,
         t_start=45,
         randomize_seed=True):

    print(model_id)
    if model_id == LDM2:
        ldm_stable = ldm2
    elif model_id == LDM2_LARGE:
        ldm_stable = ldm2_large
    else:  # MUSIC
        ldm_stable = ldm2_music

    # If the inversion was done for a different model, we need to re-run the inversion
    if not do_inversion and (saved_inv_model is None or saved_inv_model != model_id):
        do_inversion = True

    x0 = utils.load_audio(input_audio, ldm_stable.get_fn_STFT(), device=device)

    if do_inversion or randomize_seed:  # always re-run inversion
        zs_tensor, wts_tensor = invert(ldm_stable=ldm_stable, x0=x0, prompt_src=source_prompt,
                                       num_diffusion_steps=steps,
                                       cfg_scale_src=cfg_scale_src)
        wts = gr.State(value=wts_tensor)
        zs = gr.State(value=zs_tensor)
        saved_inv_model = model_id
        do_inversion = False

    # make sure t_start is in the right limit
    # t_start = change_tstart_range(t_start, steps)

    output = sample(ldm_stable, zs.value, wts.value, steps, prompt_tar=target_prompt,
                    tstart=int(t_start / 100 * steps), cfg_scale_tar=cfg_scale_tar)

    return output, wts, zs, saved_inv_model, do_inversion


def get_example():
    case = [
        ['Examples/Beethoven.wav',
         '',
         'A recording of an arcade game soundtrack.',
         45,
         'cvssp/audioldm2-music',
         '27s',
         'Examples/Beethoven_arcade.wav',
         ],
        ['Examples/Beethoven.wav',
         'A high quality recording of wind instruments and strings playing.',
         'A high quality recording of a piano playing.',
         45,
         'cvssp/audioldm2-music',
         '27s',
         'Examples/Beethoven_piano.wav',
         ],
        ['Examples/ModalJazz.wav',
         'Trumpets playing alongside a piano, bass and drums in an upbeat old-timey cool jazz song.',
         'A banjo playing alongside a piano, bass and drums in an upbeat old-timey cool country song.',
         45,
         'cvssp/audioldm2-music',
         '106s',
         'Examples/ModalJazz_banjo.wav',],
        ['Examples/Cat.wav',
         '',
         'A dog barking.',
         75,
         'cvssp/audioldm2-large',
         '10s',
         'Examples/Cat_dog.wav',]
    ]
    return case


intro = """
<h1 style="font-weight: 1400; text-align: center; margin-bottom: 7px;"> ZETA Editing 🎧 </h1>
<h2 style="font-weight: 1400; text-align: center; margin-bottom: 7px;"> Zero-Shot Text-Based Audio Editing Using DDPM Inversion 🎛️ </h2>
<h3 style="margin-bottom: 10px; text-align: center;">
    <a href="https://arxiv.org/abs/2402.10009">[Paper]</a>&nbsp;|&nbsp;
    <a href="https://hilamanor.github.io/AudioEditing/">[Project page]</a>&nbsp;|&nbsp;
    <a href="https://github.com/HilaManor/AudioEditingCode">[Code]</a>
</h3>


<p style="font-size: 0.9rem; margin: 0rem; line-height: 1.2em; margin-top:1em">
For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
<a href="https://huggingface.co/spaces/hilamanor/audioEditing?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em; display:inline" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" ></a>
</p>

"""

help = """
<b>Instructions:</b><br>
Provide an input audio and a target prompt to edit the audio. <br>
T<sub>start</sub> is used to control the tradeoff between fidelity to the original signal and text-adhearance.
Lower value -> favor fidelity. Higher value -> apply a stronger edit.<br>
Make sure that you use an AudioLDM2 version that is suitable for your input audio.
For example, use the music version for music and the large version for general audio.
</p>
<p style="font-size:larger">
You can additionally provide a source prompt to guide even further the editing process.
</p>
<p style="font-size:larger">Longer input will take more time.</p>

"""

with gr.Blocks(css='style.css') as demo:
    def reset_do_inversion():
        do_inversion = gr.State(value=True)
        return do_inversion
    gr.HTML(intro)
    wts = gr.State()
    zs = gr.State()
    saved_inv_model = gr.State()
    # current_loaded_model = gr.State(value="cvssp/audioldm2-music")
    # ldm_stable = load_model("cvssp/audioldm2-music", device, 200)
    # ldm_stable = gr.State(value=ldm_stable)
    do_inversion = gr.State(value=True)  # To save some runtime when editing the same thing over and over

    with gr.Group():
        gr.Markdown(" **note** 💡: input longer than **45 sec** is automatically trimmed")
        with gr.Row():
            input_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input Audio",
                                   interactive=True, scale=1)
            output_audio = gr.Audio(label="Edited Audio", interactive=False, scale=1)

    with gr.Row():
        tar_prompt = gr.Textbox(label="Prompt", info="Describe your desired edited output",
                                placeholder="a recording of a happy upbeat arcade game soundtrack",
                                lines=2, interactive=True)

    with gr.Row():
        t_start = gr.Slider(minimum=15, maximum=85, value=45, step=1, label="T-start (%)", interactive=True, scale=3,
                            info="Lower T-start -> closer to original audio. Higher T-start -> stronger edit.")
        # model_id = gr.Radio(label="AudioLDM2 Version",
        model_id = gr.Dropdown(label="AudioLDM2 Version",
                               choices=["cvssp/audioldm2",
                                        "cvssp/audioldm2-large",
                                        "cvssp/audioldm2-music"],
                               info="Choose a checkpoint suitable for your intended audio and edit",
                               value="cvssp/audioldm2-music", interactive=True, type="value", scale=2)

    with gr.Row():
        with gr.Column():
            submit = gr.Button("Edit")

    with gr.Accordion("More Options", open=False):
        with gr.Row():
            src_prompt = gr.Textbox(label="Source Prompt", lines=2, interactive=True,
                                    info="Optional: Describe the original audio input",
                                    placeholder="A recording of a happy upbeat classical music piece",)

        with gr.Row():
            cfg_scale_src = gr.Number(value=3, minimum=0.5, maximum=25, precision=None,
                                      label="Source Guidance Scale", interactive=True, scale=1)
            cfg_scale_tar = gr.Number(value=12, minimum=0.5, maximum=25, precision=None,
                                      label="Target Guidance Scale", interactive=True, scale=1)
            steps = gr.Number(value=50, step=1, minimum=20, maximum=300,
                              info="Higher values (e.g. 200) yield higher-quality generation.",
                              label="Num Diffusion Steps", interactive=True, scale=1)
        with gr.Row():
            seed = gr.Number(value=0, precision=0, label="Seed", interactive=True)
            randomize_seed = gr.Checkbox(label='Randomize seed', value=False)
            length = gr.Number(label="Length", interactive=False, visible=False)

    with gr.Accordion("Help💡", open=False):
        gr.HTML(help)

    submit.click(
        fn=randomize_seed_fn,
        inputs=[seed, randomize_seed],
        outputs=[seed], queue=False).then(
           fn=edit,
           inputs=[input_audio,
                   model_id,
                   do_inversion,
                   #    current_loaded_model, ldm_stable,
                   wts, zs, saved_inv_model,
                   src_prompt,
                   tar_prompt,
                   steps,
                   cfg_scale_src,
                   cfg_scale_tar,
                   t_start,
                   randomize_seed
                   ],
           outputs=[output_audio, wts, zs, saved_inv_model, do_inversion]  # , current_loaded_model, ldm_stable],
        )

    # If sources changed we have to rerun inversion
    input_audio.change(fn=reset_do_inversion, outputs=[do_inversion])
    src_prompt.change(fn=reset_do_inversion, outputs=[do_inversion])
    model_id.change(fn=reset_do_inversion, outputs=[do_inversion])
    steps.change(fn=reset_do_inversion, outputs=[do_inversion])

    gr.Examples(
        label="Examples",
        examples=get_example(),
        inputs=[input_audio, src_prompt, tar_prompt, t_start, model_id, length, output_audio],
        outputs=[output_audio]
    )

    demo.queue()
    demo.launch()
