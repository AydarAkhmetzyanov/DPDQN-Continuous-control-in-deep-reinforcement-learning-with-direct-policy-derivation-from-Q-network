if __name__ == "__main__":
    from dpdqn_v1 import DPDQN1

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import gym
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [15, 10]
    time_steps = 1e5 #testrun
    #time_steps = 100000 #for prod or even more *10?100?
    time_steps_test = int(time_steps/100)

    from stable_baselines.bench import Monitor
    from stable_baselines.results_plotter import load_results, ts2xy
    from stable_baselines import results_plotter

    os.makedirs("logs_test", exist_ok=True)
    os.makedirs("logs_train", exist_ok=True)
    os.makedirs("logs_tmp", exist_ok=True)
    from shutil import copyfile
    from utils import *


    envname = "LunarLanderContinuous-v2"
    env = gym.make(envname)
    exp_name=env.spec._env_name+'-DPDQN1'
    log_dir = 'logs_test/' + exp_name
    env = Monitor(env, log_dir, allow_early_resets=True)

    model = DPDQN1.load("models/" + log_dir.split("/")[1], env)
    obs = env.reset()
    for i in range(1000):
        if env.needs_reset:
            obs = env.reset()
        action, _states = model.predict(obs, greedy=True)
        obs, rewards, dones, info = env.step(action)
        # env.render()

    copyfile(log_dir + ".monitor.csv", "logs_tmp/tmp.monitor.csv")
    results_plotter.plot_results(["logs_tmp"], time_steps, results_plotter.X_TIMESTEPS, log_dir.split("/")[1])
    plt.show()