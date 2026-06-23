import argparse
import json  # 新增：导入 json 模块
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from openpi_client import websocket_client_policy
from tqdm import tqdm


def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="llly/piper_0621")
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    horizon = 32
    meta = LeRobotDatasetMetadata(args.repo_id)
    dataset = LeRobotDataset(
        args.repo_id,
        episodes=[args.episode_index],
        delta_timestamps={"action": [i / meta.fps for i in range(horizon)]},
    )
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # 新增：加载归一化参数中的 actions std
    with open("/home/tpy/LatentForceFlow/assets/pi05_piper_gripper2_lora/llly/piper_0621/norm_stats.json", "r") as f:
        norm_stats = json.load(f)
    action_std = np.array(norm_stats["norm_stats"]["actions"]["std"])

    loss_sum = 0.0
    norm_loss_sum = 0.0  # 新增：归一化 loss 累加器
    count = 0
    last_episode = None
    for i in tqdm(range(len(dataset)), desc=f"episode {args.episode_index}"):
        sample = dataset[i]
        episode = int(to_numpy(sample["episode_index"]))
        if episode != last_episode:
            client.reset()
            last_episode = episode

        obs = {
            "observation.images.front": to_numpy(sample["observation.images.front"]),
            "observation.images.side": to_numpy(sample["observation.images.side"]),
            "observation.images.third": to_numpy(sample["observation.images.third"]),
            "observation.state": to_numpy(sample["observation.state"]),
            "prompt": sample["task"],
        }
        pred = to_numpy(client.infer(obs)["actions"])
        gt = to_numpy(sample["action"])
        valid = ~to_numpy(sample["action_is_pad"]).astype(bool)

        n = min(len(pred), len(gt), len(valid))
        diff = pred[:n][valid[:n]] - gt[:n][valid[:n]]
        loss_sum += float(np.square(diff).sum())
        
        # 新增：计算并累加归一化后的 loss
        norm_diff = diff / action_std
        norm_loss_sum += float(np.square(norm_diff).sum())
        
        count += diff.size

    # 修改：同时打印原始 loss 和归一化 loss
    print(f"Raw Loss: {loss_sum / count}")
    print(f"Normalized Loss: {norm_loss_sum / count}")


def debug():
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    print("✅ Waiting for debugger to attach on port 5678...")
    debugpy.wait_for_client()
    print("Start to debugging")
    
if __name__ == "__main__":
    # debug()
    main()