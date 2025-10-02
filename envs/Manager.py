from envs import env_settings
from envs.UAV import UAV
from envs.env_settings import B_uav, p_uav, gamma_0, p_ue
import math
from envs.utils import distance_square_real

print_info = False


def db_to_linear(sinr_db):
    
    sinr_linear = 10 ** (sinr_db / 10)
    return sinr_linear

def get_fly_energy(uav: 'UAV'):
    """
    统一的飞行能耗计算，基于位置变化计算平均速度
    适用于离散和连续移动模式
    """
    import math
    from envs import env_settings
    
    # 计算位置变化
    if hasattr(uav, 'prev_x') and hasattr(uav, 'prev_y'):
        dx = uav.x - uav.prev_x
        dy = uav.y - uav.prev_y
        # 计算欧几里得距离作为实际移动距离
        distance = math.sqrt(dx * dx + dy * dy)
        # 平均速度 = 距离 / 时间步长
        avg_speed = distance / env_settings.step_len
    else:
        # 如果没有历史位置，使用默认速度
        avg_speed = 30
    
    # 能耗与速度平方成正比
    power = avg_speed * avg_speed / 2
    energy = power * env_settings.step_len
    
    return energy

def get_A2A_rate(uav1: 'UAV', uav2: 'UAV') -> float:

    dist_2 = distance_square_real(uav1, uav2) + 1e-6

    rate = B_uav * math.log2(1 + p_uav * uav1.trans_power_ratio * db_to_linear(gamma_0) / dist_2)

    return rate

def get_A2G_rate(uav1: 'UAV',task: 'Task') -> float:

    dist_2 = distance_square_real(uav1, task) + 1e-6

    rate = B_uav * math.log2(1 + p_ue * db_to_linear(gamma_0) / dist_2)

    return rate

def get_trans_task_time_and_energy(uav1: UAV, uav2: UAV, task: 'Task'):
    rate = get_A2A_rate(uav1, uav2)
    trans_time = task.ok / rate
    trans_energy = trans_time * p_uav * uav1.trans_power_ratio
    trans_time = trans_time // env_settings.step_len
    
    
    return int(trans_time), trans_energy


def get_take_task_time_and_energy(uav: UAV, task: 'Task'):
    rate = get_A2G_rate(uav, task)
    take_time = task.ok / rate
    take_energy = take_time * p_ue
    take_time = take_time // env_settings.step_len
    if print_info:
        print(f'{task.name}, take_time:{take_time}, rate:{rate}, energy:{take_energy}')
    return int(take_time), take_energy


def get_uav_execute_task_time_and_energy(uav: UAV, task: 'Task'):
    computing_rate = uav.computing_rate * uav.computing_ratio
    compute_delay = env_settings.ok_from * task.fk / computing_rate
    compute_energy = env_settings.kappa_0 * pow(computing_rate * 1e6, 3) * compute_delay

    compute_delay = compute_delay // env_settings.step_len
    if print_info:
        print(f'{task.name}, execute_time:{compute_delay}, energy:{compute_energy}')
    return int(compute_delay), compute_energy


def get_min_execute_time():
    uav = UAV('test-1')
    computing_rate = uav.computing_rate
    compute_delay = env_settings.ok_from * env_settings.fk_to/ computing_rate // env_settings.step_len
    return compute_delay



if __name__ == '__main__':
    print(get_min_execute_time())