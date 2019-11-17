if __name__ == "__main__":
    from dpdqn_v2 import DPDQN2

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import gym
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [15, 10]
    time_steps = 1e4 #testrun
    #time_steps = 100000 #for prod or even more *10?100?
    time_steps_test = int(time_steps/10)

    from stable_baselines.bench import Monitor
    from stable_baselines.results_plotter import load_results, ts2xy
    from stable_baselines import results_plotter

    os.makedirs("logs_test", exist_ok=True)
    os.makedirs("logs_train", exist_ok=True)
    os.makedirs("logs_tmp", exist_ok=True)
    os.makedirs("logs_dev", exist_ok=True)
    from shutil import copyfile
    from utils import *





    #train
    envname = "Pendulum-v0"
    envname = "LunarLanderContinuous-v2"
    envname = "BipedalWalker-v2"
    env = gym.make(envname)
    exp_name=env.spec._env_name+'-DPDQN2'

    log_dir='logs_dev/'+exp_name
    env = Monitor(env, log_dir, allow_early_resets=True)

    model = DPDQN2(env, verbose=1)

    print("time_steps_todo: "+str(time_steps))
    model.learn(total_timesteps=int(time_steps))
    copyfile(log_dir+".monitor.csv", "logs_tmp/tmp.monitor.csv")
    results_plotter.plot_results(["logs_tmp"], time_steps, results_plotter.X_TIMESTEPS, log_dir.split("/")[1])
    plt.show()






    os.makedirs("models", exist_ok=True)
    model.save("models/"+log_dir.split("/")[1])




    # test

    env = gym.make(envname)
    log_dir = 'logs_test/' + exp_name
    env = Monitor(env, log_dir, allow_early_resets=True)

    model = DPDQN2.load("models/" + log_dir.split("/")[1], env)
    obs = env.reset()
    for i in range(time_steps_test):
        if env.needs_reset:
            obs = env.reset()
        action, _states = model.predict(obs, greedy=False)
        obs, rewards, dones, info = env.step(action)
        #env.render()

    copyfile(log_dir + ".monitor.csv", "logs_tmp/tmp.monitor.csv")
    results_plotter.plot_results(["logs_tmp"], time_steps, results_plotter.X_TIMESTEPS, log_dir.split("/")[1])
    plt.show()

    # test greedy

    env = gym.make(envname)
    log_dir = 'logs_test/' + exp_name
    env = Monitor(env, log_dir, allow_early_resets=True)

    model = DPDQN2.load("models/" + log_dir.split("/")[1], env)
    obs = env.reset()
    for i in range(time_steps_test):
        if env.needs_reset:
            obs = env.reset()
        action, _states = model.predict(obs, greedy=True)
        obs, rewards, dones, info = env.step(action)
        #env.render()

    copyfile(log_dir + ".monitor.csv", "logs_tmp/tmp.monitor.csv")
    results_plotter.plot_results(["logs_tmp"], time_steps, results_plotter.X_TIMESTEPS, log_dir.split("/")[1])
    plt.show()


