from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

model_cache = {}


def download_model():
    model_name = "llava-v1.5-7b-Q4_K_M.gguf"
    repo_id = "second-state/Llava-v1.5-7B-GGUF"
    local_path = os.path.join(os.path.dirname(__file__), model_name)

    if not os.path.exists(local_path):
        print(f"Downloading {model_name} from Huggingface...")
        local_path = hf_hub_download(
            repo_id=repo_id, filename=model_name, local_dir="src/models"
        )
    return local_path


def get_model(modelpath):
    if modelpath not in model_cache:
        model_cache[modelpath] = Llama(
            model_path=modelpath,
            n_ctx=1024,
            n_gpu_layers=10,  # Adjust based on GPU VRAM
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

    output = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=128,
    )

    return output["choices"][0]["message"]["content"]
