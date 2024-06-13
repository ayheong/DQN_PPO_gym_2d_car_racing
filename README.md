## Car Racing with DQN and PPO
This project demonstrates the implementation of two reinforcement learning algorithms, Deep Q-Network (DQN) and Proximal Policy Optimization (PPO), to solve the CarRacing-v2 environment from Gymnasium.

### Dependecies
Make sure to install the required dependencies (some general ones are listed below):
- Gymnasium (0.29.1)
- Box2D (2.3.5)
- PyTorch (2.3.0)
- TorchVision (0.18.0)
- Torchaudio (2.3.0)
- NumPy (1.26.4)
- OpenCV (4.9.0)
- Matplotlib (3.8.4)
- CUDA (12.1)
- cuDNN (8)

### Running the train and test scripts
To train the DQN agent, navigate to the DQN folder and run:
```bash
python test.py
```

To test a saved DQN agent model, navigate to the DQN folder, modify the file name paramater in the test.py code to the name of the saved model and run:
```bash
python test.py --test
```

To train the PPO agent, navigate to the PPO folder and run:
```bash
python ppo_test.py
```

To test a saved PPO agent model, navigate to the PPO folder, modify the file name paramater in the ppo_test.py code to the name of the saved model and run:
```bash
python ppo_test.py --test
```
