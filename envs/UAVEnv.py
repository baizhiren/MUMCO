from envs.discrete_uav_scenario import Scenario
from envs.environment import MultiAgentEnv
import envs.settings as settings

def UAVEnv(**kwargs):
    

    
    scenario = Scenario()
    
    world = scenario.make_world(**kwargs)

    scenario_reward = scenario.shape_reward if settings.reward_shaping else scenario.reward
    env = MultiAgentEnv(world, scenario.reset_world,
                        scenario_reward, scenario.observation, scenario.info)

    return env



if __name__ == '__main__':
    print('hello')
    env = UAVEnv()




