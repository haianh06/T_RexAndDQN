import os
import cv2
import torch
import numpy as np
from DinoEnv import DinoEnv
from DQNAgent import DQNAgent

# === Cấu hình ===
episodes = 1000
save_path = "./dqn_trex_modelv2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Khởi tạo môi trường và agent ===
env = DinoEnv()
obs_shape = (1, 84, 84)  # 1 channel grayscale
n_actions = env.action_space.n
agent = DQNAgent(input_shape=obs_shape, num_actions=n_actions, device=device)

# === Huấn luyện ===
for ep in range(1, episodes + 1):
    state = env.reset()
    total_reward = 0
    done = False
    frame_count = 0

    print(f"\n[EPISODE {ep}] Starting...")

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        #print(f"Step {frame_count:03d} | Action: {action} | Reward: {reward:.2f} | Done: {done}")

        agent.remember(state, action, reward, next_state, done)
        agent.train_step()

        state = next_state
        total_reward += reward
        frame_count += 1

        # Hiển thị (debug / kiểm tra mắt)
        # if ep % 50 == 0:
        #     cv2.imshow("Training", state.squeeze())
        if cv2.waitKey(1) == 27:
            break

    # Cập nhật target model mỗi 10 episodes
    if ep % 10 == 0:
        agent.update_target()
        print("[Target model updated]")

    print(f"[EPISODE {ep} DONE] Total reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f} | Frames: {frame_count}")

    # Lưu mô hình
    if ep % 50 == 0:
        torch.save(agent.model.state_dict(), save_path)
        print(f"[Saved model at {save_path}]")

env.close()
cv2.destroyAllWindows()