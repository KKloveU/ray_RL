import ray
import torch
from models import Model
import numpy as np
import random
from atari_wrappers import make_atari, wrap_deepmind


@ray.remote
class Player:
    def __init__(self, checkpoint, replay_buffer, share_storage, test_mode):
        self.game = make_atari(checkpoint["game"]+"NoFrameskip-v4")
        self.game = wrap_deepmind(self.game, scale=True, frame_stack=True)
        self.action_list = checkpoint["action_list"]
        self.epsilon = checkpoint['epsilon']
        self.max_len_episode = checkpoint["max_len_episode"]
        self.checkpoint = checkpoint
        self.training_step = checkpoint["training_step"]
        self.epsilon_decay = checkpoint['episilon_decay']
        self.update_memory_iter = checkpoint['update_memory_iter']

        self.replay_buffer = replay_buffer
        self.share_storage = share_storage

        self.model = Model().cuda()
        self.model.set_weights(checkpoint["weights"])

        self.test_mode = test_mode
        if self.test_mode:
            self.palyed_game = 0
            self.epr_writer = open('./log/'+checkpoint["game"]+'.log', 'w')

        print('player init done')

    def continous_self_play(self):
        print('start play')
        while not ray.get(self.share_storage.get_info.remote("terminate")):
            self.model.set_weights(
                ray.get(self.share_storage.get_info.remote('weights')))
            self.paly_game()
            if ray.get(self.share_storage.get_info.remote("start_training")) == True:
                self.epsilon = self.epsilon*(1-self.epsilon_decay)

        if self.test_mode:
            self.epr_writer.close()
        print('end play')

    def paly_game(self):
        game_history = GameHistory()
        ep_r = 0
        done = False
        obs = self.game.reset()
        self.game.step(1)
        step = 0
        live = 5

        with torch.no_grad():
            while not done:
                fake_done = False
                action_index = self.choose_action(np.array(obs))
                obs_, reward, done, info = self.game.step(
                    self.action_list[action_index])
                if info["ale.lives"] != live:
                    reward = -1
                    live = info["ale.lives"]
                    self.game.step(1)
                    fake_done = True
                if not self.test_mode:
                    game_history.save_transition(
                        np.array(obs), action_index, reward, np.array(obs_), fake_done)
                    if done or step % self.update_memory_iter == 0:
                        self.replay_buffer.store_memory.remote(game_history)
                        game_history.clear_memory()
                obs = obs_
                ep_r += reward
                step += 1

            if self.test_mode:
                print(self.palyed_game, ep_r, self.epsilon)
                self.epr_writer.write(str(ep_r)+'\n')
                self.epr_writer.flush()
                self.palyed_game += 1

    def choose_action(self, obs):
        if self.test_mode:
            obs_input = torch.FloatTensor(
                obs).cuda().permute(2, 0, 1).unsqueeze(0)
            action_index = np.argmax(self.model(obs_input).cpu().numpy()[0])
            return action_index

        if random.random() > self.epsilon:
            obs_input = torch.FloatTensor(
                obs).cuda().permute(2, 0, 1).unsqueeze(0)
            action_index = np.argmax(self.model(obs_input).cpu().numpy()[0])
        else:
            action_index = random.randint(0, len(self.action_list)-1)
        return action_index


class GameHistory:
    def __init__(self) -> None:
        self.trans_history = []

    def save_transition(self, obs, a, r, obs_, done):
        self.trans_history.append([obs, a, r, obs_, done])

    def clear_memory(self):
        self.trans_history = []
