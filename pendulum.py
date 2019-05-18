import time
import gym
env = gym.make('Pendulum-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()

        print(observation)
        #action = [0]
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(action)
        print(env.step(action))
        time.sleep(1)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()
# print(env.action_space)
# print(env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.high)
# print(env.action_space.low)
