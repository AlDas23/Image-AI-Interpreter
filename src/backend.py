import base64
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from huggingface_hub import hf_hub_download
import os

model_cache = {}


def image_to_base64_data_uri(file_path):
    with open(file_path, "rb") as img_file:
        base64_data = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/png;base64,{base64_data}"


def download_model():
    model_name = "ggml-model-q4_k.gguf"
    repo_id = "mys/ggml_llava-v1.5-7b"
    local_path = os.path.join(os.path.dirname(__file__), model_name)

    if not os.path.exists(local_path):
        print(f"Downloading {model_name} from Huggingface...")
        local_path = hf_hub_download(
            repo_id=repo_id, filename=model_name, local_dir="models"
        )
    return local_path


def get_model(modelpath):
    if modelpath not in model_cache:
        chat_handler = Llava15ChatHandler.from_pretrained(
            repo_id="mys/ggml_llava-v1.5-7b",
            filename="*mmproj*",
        )
        model_cache[modelpath] = Llama(
            model_path=modelpath,
            n_ctx=2048,
            chat_handler=chat_handler,
            n_gpu_layers=-1,
            logits_all=False,
            verbose=True,
        )

    return model_cache[modelpath]


def generate_response(modelpath, prompt, img):

    if not modelpath or modelpath == "None":
        modelpath = download_model()
    else:
        modelpath = "models/" + modelpath

    llm = get_model(modelpath)

    img64 = image_to_base64_data_uri(img)

    output = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are an assistant who perfectly describes images. Carefully analyze the image and provide a detailed and correct answer to the user's question.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img64}},
                ],
            },
        ]
    )

    return output["choices"][0]["message"]["content"]
