# ray_RL

本项目主要有两条分支,一条是基于DQN的多种优化方法的改进,使用了DDQN,Dueing-DQN,Noisy-DQN,Prioritized-DQN的优化方法。
一条是基于Policy-base的改进方法,实现了基于ray的分布式PPO算法。

算法主要分为四个基本结构,Player(数据采样),Replay_Buffer(经验缓冲池),Trainer(学习),Share_Storage(策略存放).
算法执行过程为，使用多个Player同步采集数据。每个Play定期从Share_Storage中获取最新的网络参数，并将采集到经验数据存放在Replay_Buffer中。
使用Trainer更新网络参数。Trainer每次从Replay_Buffer中采样一个batch的数据，用于网络参数的更新，并定期将最新的网络参数存放在Share_Storage中。

# 使用方法

1、将代码在每台机器人上进行拷贝。
2、修改为合适的参数后，在其中一台机器上执行 python main.py
3、之后在其余的节点电脑上执行 ray start --address=<ip>:<port> 
即可调度多台电脑执行程序，其中针对不同的集群需要自行调节 main.py 中 options(num_cpus=<num_cpu>,num_gpus=<num_gpu>)为不同的模块分配合适的计算资源。
4、运行结束后，需要在非主节点运行 ray stop


# 修改参数

算法的所有参数均配置在 main.py 文件的 checkpoint字典中。
game 为 OpenAI gym Atari 游戏名
action_list 为该游戏的动作空间
num_cluster 为采集者分组
num_wrokers 为每组采集者的数量，即 Player = num_cluster * num_workers
注意，需要将同一组的Player设置在同一台物理计算节点，以减少通信带宽消耗。



