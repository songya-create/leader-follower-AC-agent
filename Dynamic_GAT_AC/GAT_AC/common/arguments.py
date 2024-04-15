import argparse

"""
Here are the param for the training

"""


def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--seq_len", type = int, default =1 , help = "seq_len")
    parser.add_argument("--agents_num", type=int, default=12, help="set number of my agent")
    parser.add_argument("--edge_n", type=int, default=2,help="set number of my agent")
    parser.add_argument("--algorithm",type=str,default="CommCritic",help="algorithm_use")
    # Environment
    parser.add_argument("--scenario-name", type=str, default="simple_spread_po", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=400, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=4000000, help="number of time steps")
    # 一个地图最多env.n个agents，用户可以定义min(env.n,num-adversaries)个敌人，剩下的是好的agent
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--shape", type=int, default=0, help="number of adversaries")
    # Core training parameters
    parser.add_argument("--lr-actor", type=float, default=1e-3, help="learning rate of actor")#5e-4
    parser.add_argument("--lr-critic", type=float, default=1e-3, help="learning rate of critic")#1e-3
    parser.add_argument("--epsilon", type=float, default=0, help="epsilon greedy")#0.4
    parser.add_argument("--noise_rate", type=float, default=0, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.001, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(2e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256  , help="number of episodes to optimize at the same time")


    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=47000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")

    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=2, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=400, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=True, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=500000, help="how often to evaluate model")

    args = parser.parse_args()

    return args
