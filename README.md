# CarRacing-mod
Creating a highly customizable CarRacing environment under openai.gym format

This environment provides a modified openAI Gym CarRacing environment allowing a series of "world" and Reward function customizations. The idea is to offer a series of tools to quickly test different reinforcement learning strategies without having to deal with the world programming.

The four car_racing___.py files define different world environments, and are registered to gym:

CarRacing-v0 is the original implementation

CarRacing-v1 only changes the action space to a 5 actions discrete set (see below for details)

CarRacing-v2 is my modified version including discrete actions, reward functions, frame_stacking, frame_skipping, track reuse, starting points, track geometry, indicators, and more

CarRacing-v3 is an upcoming version including random objects on track.

# Installation



# Usage


# Examples
