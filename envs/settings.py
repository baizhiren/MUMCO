env = 'test'

algorithm = 'rl'



task_num = 10
uav_num = 10
hop_limit = 200

obs_tasks_num = 3
uav_to_uav_dist = 10
uav_to_task_dist = 3
task_alive_time = 30
length = 20
width = 20
size_length = 50
world_length = 100
render_gap = 0.5
small_task_num_range = [2, 4]
frozen_time = 3
bigTask_ratio = 0.3333333333333333

print_success_or_fail = False
print_reward_detail = False
jump_render = True
skip_before_render_num = 0
debug = True
print_step = False
done_factor = 10
trans_energy_factor = 0
compute_energy_factor = 10
print_last_only_enable = True
print_last_only = True
w1 = 40
w2 = 1
fail_delay = 40

use_continuous_movement = False
reward_shaping = True

enable_animated_trajectory = False
fix_trans_and_compute_when_train = False
use_gaussian_task_generation = True
gaussian_centers = [(15, 3), (10, 15), (3, 3)]
gaussian_weights = [1, 2, 1]
gaussian_sigmas = [1.5, 2, 1.5]
