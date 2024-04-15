from runner import Runner
from common.arguments import get_args
from common.utils import make_env
import numpy as np
import random
import torch


if __name__ == '__main__':
    # get the params
    args = get_args()
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate==True:
          runner.args.evaluate_episodes=50
          runner.evaluate()

    else:
        runner.run()
        while(1):
            continue_or=input("\nif continue trainning？please enter yes or no or eva\n")
            if continue_or=='yes':
                runner.run()
            elif continue_or=='eva':
                runner.evaluate()
            else:
                break
        print('ending test begin ----------1000times')
       # runner.evaluate1000()
