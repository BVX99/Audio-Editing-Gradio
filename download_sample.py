import subprocess, os
assets_folder = "./Examples/"
if not os.path.exists(assets_folder):
    os.makedirs(assets_folder)
files = {
    "Examples/Beethoven.wav":"https://huggingface.co/spaces/hilamanor/audioEditing/resolve/main/Examples/Beethoven.wav",
    "Examples/Beethoven_arcade.wav":"https://huggingface.co/spaces/hilamanor/audioEditing/blob/main/Examples/Beethoven_arcade.wav",
    "Examples/Beethoven_piano.wav":"https://huggingface.co/spaces/hilamanor/audioEditing/blob/main/Examples/Beethoven_piano.wav",
    "Examples/Cat.wav":"https://huggingface.co/spaces/hilamanor/audioEditing/blob/main/Examples/Cat.wav",
    "Examples/Cat_dog.wav":"https://huggingface.co/spaces/hilamanor/audioEditing/blob/main/Examples/Cat_dog.wav",
    "Examples/ModalJazz.wav":"https://huggingface.co/spaces/hilamanor/audioEditing/blob/main/Examples/ModalJazz.wav",
    "Examples/ModalJazz_banjo.wav":"https://huggingface.co/spaces/hilamanor/audioEditing/blob/main/Examples/ModalJazz_banjo.wav",
}
for file, link in files.items():
    file_path = os.path.join(assets_folder, file)
    if not os.path.exists(file_path):
        try:
            subprocess.run(['wget', link, '-O', file_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {file}: {e}")
