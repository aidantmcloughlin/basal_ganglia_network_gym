from gym.envs.registration import register

def register_envs():
    register(
        id='bg-network-v0',
        entry_point='env.bg_network:BGNetwork'
    )
