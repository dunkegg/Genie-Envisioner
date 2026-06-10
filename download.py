# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id="nvidia/Cosmos-Predict2-2B-Video2World",
#     local_dir="checkpoints/Cosmos-Predict2-2B-Video2World",
#     max_workers=1,
#     resume_download=True,
#     # request_timeout=60,  
# )
import time
from huggingface_hub import snapshot_download

# from modelscope.hub.snapshot_download import snapshot_download


for i in range(20):
    try:
        # snapshot_download(
        #     repo_id="Lightricks/LTX-Video",
        #     local_dir="checkpoints/LTX-Video",
        #     max_workers=1,
        # )
        snapshot_download(
            repo_id="nvidia/Cosmos-Predict2-2B-Video2World",
            local_dir="checkpoints/Cosmos-Predict2-2B-Video2World",
            # max_workers=1,
            resume_download=True,
            # request_timeout=60,  
        )

        # from modelscope import snapshot_download
        # snapshot_download(
        #     'agibot_world/Genie-Envisioner',
        #     local_dir='checkpoints/Genie',
        #     max_workers=8,
        # )

        print("download complete")
        break

    except Exception as e:
        print(f"retry {i}: {e}")
        time.sleep(3)


