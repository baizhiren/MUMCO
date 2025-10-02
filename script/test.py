#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from datetime import datetime

import torch

from onpolicy.config import get_config

from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from onpolicy.envs.uavs import settings, ans
from onpolicy.envs.uavs.rendering import plot_uav_trajectories, plot_uav_trajectories_animated, test_trajectory_plotting
from onpolicy.envs.uavs.UAVEnv import UAVEnv
from onpolicy.runner.shared.uav_runner import UAVRunner as Runner


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "UAV":
                env = UAVEnv()
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")
    parser.add_argument('--run', type=int, default=0, help="run的标号，快速启动")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def read_field_from_file(file_path, field_name):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # 读取文件所有行
        for i in range(len(lines)):
            line = lines[i].strip()
            if line == f"--{field_name}":  # 找到字段名
                # 下一行是该字段的值
                if i + 1 < len(lines):
                    return lines[i + 1].strip()  # 返回字段的值
    return None  # 如果未找到该字段，返回 None


def convert_value(value, value_type):
    """根据字段的目标类型转换值"""
    if value_type == int:
        return int(value)
    elif value_type == float:
        return float(value)
    elif value_type == bool:
        return value.lower() == 'true'
    else:
        return value  # 默认情况（假设类型为 str）


def update_args_from_file(run_dir, field_names: list, all_args,
                          parser):
    """根据配置文件更新参数"""
    for name in field_names:
        # 读取字段值
        value = read_field_from_file(run_dir / 'args_output.txt', name)

        if value is not None:
            # 获取该字段在 parser 中定义的类型
            field_type = None
            for action in parser._actions:
                if action.dest == name:
                    field_type = action.type
                    break

            if field_type is None:
                raise ValueError(f"Type for field '{name}' not found in parser.")

            # 转换为正确的类型
            converted_value = convert_value(value, field_type)
            # 设置字段值
            setattr(all_args, name, converted_value)


def main(args):
    settings.env = 'render'

    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    assert all_args.use_render, ("u need to set use_render be True")
    # assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_rollout_threads == 1, ("only support to use 1 env to render.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    cur_run = None
    for folder in run_dir.iterdir():
        if int(str(folder.name).split('run')[1]) == all_args.run:
            cur_run = str(folder.name)
    settings.current_run = 'run' + str(all_args.run)
    run_dir = run_dir / cur_run
    all_args.model_dir = run_dir / 'models'

    field_names = ['hidden_size', 'layer_N']
    update_args_from_file(run_dir, field_names, all_args, parser)

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    if settings.algorithm == 'rl' or settings.algorithm == 'OHM':
        runner.render()
    else:
        runner.render_other_algorithm()

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])

