import torch
import cv2
import time
from DinoEnv import DinoEnv
from DQNAgent import DQNAgent
# Thiết lập thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo môi trường và agent
env = DinoEnv()
obs_shape = (1, 84, 84)
n_actions = env.action_space.n
agent = DQNAgent(input_shape=obs_shape, num_actions=n_actions, device=device)

# Load mô hình đã train
model_path = "./dqn_trex_modelv2.pth"
agent.model.load_state_dict(torch.load(model_path, map_location=device))
agent.model.eval()
agent.epsilon = 0.0  # Không random, luôn dùng mạng

# Chạy mô hình
state = env.reset()
done = False
total_reward = 0

print("Bắt đầu chạy agent đã huấn luyện... (ấn ESC để thoát)")

while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    total_reward += reward

    # # Hiển thị quan sát
    # img = state.squeeze()
    # img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # cv2.putText(img_bgr, f"Reward: {total_reward:.1f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    # cv2.imshow("Test Agent", img_bgr)
    if cv2.waitKey(10) == 27:  # ESC để thoát
        break

print(f"Tổng reward: {total_reward:.2f}")

env.close()
cv2.destroyAllWindows()
