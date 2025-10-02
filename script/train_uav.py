
import sys
import os
import types

import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs import settings
from envs.UAVEnv import UAVEnv

from envs.env_wrappers import SubprocVecEnv, DummyVecEnv


settings.algorithm = 'rl'



class DualWriter:
    def __init__(self, file):
        self.console = sys.stdout  
        self.file = open(file, 'a')  

    def write(self, message):
        self.console.write(message)  
        self.file.write(message)     

    def flush(self):
        
        self.console.flush()
        self.file.flush()




def make_train_env(all_args):
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


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "UAV":
                env = UAVEnv()
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    settings.env = "train"
    settings.print_last_only = True
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
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        
        all_args.use_centralized_V = False
    else:
        pass

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    
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

    
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir()]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)

        settings.current_run = curr_run

        from datetime import datetime

        
        now = datetime.now()

        
        formatted_time = now.strftime("%Y-%m-%d_%H-%M")

        run_dir = run_dir / f"{formatted_time}-uav{settings.uav_num}-task{settings.task_num}-layer{all_args.layer_N}-hidden{all_args.hidden_size}-map{settings.length}x{settings.width}-lr{all_args.lr}-{curr_run}"
        print(f"run dir:{run_dir}")
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    
    sys.stdout = DualWriter(f'{run_dir}/train_log.txt')

    variables = globals().copy()
    with open(f'{run_dir}/variables_output.txt', 'w') as f:
        
        for var_name, var_value in variables.items():
            if '__' in var_name or callable(var_value) or isinstance(var_value, types.ModuleType):
                continue
            f.write(f"{var_name} = {var_value}\n")

        
    print("全局变量已写入 variables_output.txt 文件")


    

    
    with open(f'{run_dir}/args_output.txt', 'w') as f:
        
        for arg in args:
            f.write(f"{arg}\n")
            
        print("命令行参数已写入 args_output.txt 文件")




    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    
    
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    
    
    
    from runner.shared.uav_runner import UAVRunner as Runner
    
    
    runner = Runner(config)
    runner.run()
    
    
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
