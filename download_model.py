from huggingface_hub import snapshot_download

if __name__ == '__main__':
    # model_id="lmms-lab/LLaVA-NeXT-Video-7B-DPO"
    # model_id = "lmms-lab/llava-next-interleave-qwen-7b-dpo"
    model_id = "lmms-lab/llava-next-interleave-qwen-0.5b"
    snapshot_download(
        repo_id=model_id,
        local_dir=f"models/{model_id}",
        local_dir_use_symlinks=False,
        revision="main"
    )

