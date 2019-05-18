import time
import gym
import numpy
import random
import math
env = gym.make('Pendulum-v0')
#stałe:
liczeb_y = 20  # liczebnosc y
liczeb_x = 20  # liczebnosc x
liczeb_m = 32  # liczebnosc momentow
liczeb_r = 24  # liczebnosc reward

liczeb_a = 20  # liczebnosc action

# Init arbitary values
q_table = numpy.full((liczeb_x,liczeb_y,liczeb_m,liczeb_a),0)

alpha = 0.1
gamma = 0.6
epsilon = 0.1

def aprox(number, amount,max):
    posit = int(math.floor(((number/max)+1)*amount/2))
    return posit



for i_episode in range(100):
    state = env.reset()

    for t in range(100):
        statey = aprox(state[0],liczeb_y,1)
        statex = aprox(state[1],liczeb_x,1)
        statem = aprox(state[2],liczeb_m,8)


        if random.uniform(0, 1) < epsilon:
            # Check the action space
            action = env.action_space.sample()
        else:
            # Check the learned values
            action = numpy.array([float(numpy.argmax(q_table[statey,statex,statem])-10)/5])

        next_state, reward, done, info = env.step(action)

        nstatey = aprox(next_state[0], liczeb_y, 1)
        nstatex = aprox(next_state[1], liczeb_x, 1)
        nstatem = aprox(next_state[2], liczeb_m, 8)


        old_value = q_table[statey,statex,statem, aprox(action[0],liczeb_a,2)]
        next_max = numpy.max((q_table[q_table[nstatey,nstatex,nstatem]]-10)/5)


        # Update the new value
        new_value = (1 - alpha) * old_value + alpha * \
                    (reward + gamma * next_max)

        q_table[statey,statex,statem, aprox(action[0],liczeb_a,2)] = new_value

        print(next_state)

        state = next_state

        env.render()
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




