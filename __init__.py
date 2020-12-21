try:
    import Box2D
    from gym.envs.box2d.lunar_lander import LunarLander
    from gym.envs.box2d.lunar_lander import LunarLanderContinuous
    from gym.envs.box2d.bipedal_walker import BipedalWalker, BipedalWalkerHardcore
    from gym.envs.box2d.car_racing_old import CarRacing
    from gym.envs.box2d.car_racing_discre import CarRacing as CarRacing1
    from gym.envs.box2d.car_racing_pab import CarRacing2
    from gym.envs.box2d.car_racing_obst import CarRacing2 as CarRacing3
except ImportError:
    Box2D = None
