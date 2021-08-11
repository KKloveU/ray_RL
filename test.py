import gym
import random
import time
env=gym.make('Pong-v0')
env.reset()
for i in range(30):
    a=random.randint(0,3)
    print(a)
    env.step(5)
    env.render()
    time.sleep(0.1)