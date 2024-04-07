from datetime import datetime
import os
import torch

day_time = datetime.now().strftime("%Y-%m-%d_%H_%M")


def save_model(model, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        os.path.join(path, day_time + ".pth"),
    )
    # 保存模型的状态字典
    torch.save(
        {
            "model_state_dict": model.state_dict(),
        },
        os.path.join(path, "latest.pth"),
    )
