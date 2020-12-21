# CarRacing-mod
Creating a *highly customizable* CarRacing environment under openai.gym environments format

This environment provides a modified openAI Gym CarRacing environment allowing a series of "world" and Reward Function customization. The idea is to offer you a set of tools to quickly test different Reinforcement Learning strategies without having to deal with all the world programming.

The four `car_racing_xxx.py` files included define different world environments, and are all already registered to gym via both `__init__.py` files provided:

* CarRacing-v0 is the original openAI implementation (*), included for back-to-back tests and for challenge participation

* CarRacing-v1 introduces a small but *fundamental* change to the original: the action space is modified to a discrete one, with only 5 possible actions allowed (see below for details)

* CarRacing-v2 is my own modified version of the environment, allowing the possibility of including:
	* discrete actions, customizable 
	* customizable reward function 
	* frame_stacking and frame_skipping 
	* selectable frame colors/channels
	* same track reuse and progressive starting points (DeepRacer style)
	* track geometry control, although kind of limited
	* indicators bar hiding
	* and more

* CarRacing-v3 is an upcoming version that will include randomly placed objects on track, that need to be avoided

(*) **Note:** From *gym version 0.17.3* onward, openAI provides a pyglet optimized environment, running up to 50% faster, as tested on my hardware. Although, I am experimenting memory leak problems with that environment, so I decided to roll back to the original (slower) version for now.

# Installation

It is highly recommendable that you start your toying with a new environment. Python 3.6 will do well.
Then `pip install gym`, following on Linux distro with `pip install 'gym[box2d]'  #all dependencies for CarRacing` or on Win10 (preferably with WSL2 enabled!) execute `conda install -c conda-forge box2d-py ` from a terminal with the right environment activated.

As reinforcement learning kit I have been using *Stable baselines*. Its a great project with a very friendly documentation. Beware of TensorFlow version limitations with this toolkit.
```
pip install stable-baselines
pip install tensorflow==1.15.0
```
Feel free to use the RL tools you like, just make sure to understand the mechanics behind the *gym environments*

Once finished installing, you will need to manually copy/overwrite the provided files. 
Sorry for that, but is way faster than any wheel implementation...
* locate your environment location on disk. Usually under `./anaconda/envs/`
* search for `car_racing.py`
* open file location, this will be a `../gym/envs/box2d` directory
* copy and **overwrite** all provided files
* move up one directory to `../gym/envs`
* copy and **overwrite** the `__init__` file provided on the repo ENVS directory
* run the provided test notebook `test_gym_mod_installation`
* if you don't see 2 classical environments and **3 times** a window popping up with a car driving clumsily, something went wrong! 
* Try copying the files again. Rest assure you *should* have not broken anything else on your gym installation


# Usage

CarRacing-v2 and -v3 provide a set of customization parameters, as follows:
```
env = gym.make('CarRacing-v2',
        use_track = 1,           # number of times to use the same track, [1-100]. More than 20 high risk of overfitting!!
        episodes_per_track = 1,  # number of evenly distributed starting points on each track [1-20]. Every time you call reset(), the env automatically starts at the next point
        discre = ACT,            # Action discretization function, None for continous
        tr_complexity = 12,      # generated track geometric complexity, [6-20]
        tr_width = 40,           # relative track width, [30-50]
        patience = 2,            # Max time in secs without progress, [0.5-10]
        indicators = True,       # Show or not bottom info panel
        game_color = 0,          # State (frame) color option: 0 = RGB, 1 = Grayscale, 2 = Green only
        frames_per_state = 1,    # stacked (rolling history) frames on each state [1-inf], latest observation always on first frame
        skip_frames = 0,         # number of consecutive frames to skip on history [0-4]
        f_reward = STD_REWARD,   # reward function coefficients, refer to Docu for details
        verbose = 1      )
```
The user can replace ACT and STD_REWARD with custom np.arrays, in order to adapt *discre* and *f_reward* to his needs.

### Track generation and complexity
Each track starts as a full circle of predetermined radius, and gets divided in `tr_complexity` equal angular segments. Then these segments are radially randomly moved from 1/3 to 1 of radius. Finally a loop connects the dots, fits curves and retries until some geometric error allowance is achieved.

The resulting track perimeter gets then divided in *n* equal length tiles, meaning the (at track generation time) informed `n_tiles` is a good proxy for track length.

### Real car action space
After thoroughly investigating `car_dynamics.py`, is my understanding this model will really work BADLY in continuous mode...
Physics model control actions, can be simultaneous => steering_angle, gas (throttle), brake_level, but the model is friction limited @1
Due to `car_dynamics.py` setup, 
* d(gas)/dt is limited to 0.1 per frame, so it takes 10 steps to get throttle to max 
* braking >= 0.9 blocks wheels immediately, otherwise reduces angular speed an all wheels
* the car is friction limited @1 currently, so simultaneous steer and brake is limited
* beware steering actually saturates @+-0.4 
* it takes 7 steps to fully turn wheels either side from center @ steering >= 0.4, steering rate is 0.06/step
* the car presents lots of wheel-spin. Throttle < 1 might be a faster way to accelerate the car

Other than the DeepRacer model, the throttle action here gives you an acceleration, not a target velocity!!
--> it makes less sense to have many throttle levels

### Discretization
For starters, the simplest option would be
`ACT = [[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.8]]`
discrete actions: center_steering and no gas/brake, left, right, accel, brake  
     --> actually a good choice, because *car_dynamics* softens the action's diff for gas and steering

`ACT2= [[0, 0, 0], [-0.4, 0, 0], [0.4, 0, 0], [0, 0.8, 0], [0, 0, 0.8], 
[-0.2, 0, 0], [0.2, 0, 0], [0, 0.4, 0], [0, 0, 0.4]]`
ACT2 is only useful for actually driving outside the track without drifting around
discrete actions: center_steering and no gas/brake, left, right, accel, brake, half_left, half_right, half_accel, soft_brake

### REWARDS 
For this environment I propose a fixed, parameterized reward function --> no scripting required

By selecting 9 parameters you can change the reward function behavior, as follows:
`SR = [a, b, c, d, e, f, g, h, i]`

After each step: `reward = a + b * dist_center + c * car_vel_mag + d * steer_angle` representing penalties for: 
* each step taken 
* distance to track's centerline, `dist_center = (1 - dist_to_nearest_waypoint / track_width)`, clipped to +-1
* car lineal speed magnitude 
* car wheel steer angle (not necessary the steering input!)

Each time a new tile is reached: `reward = reward + e * 100 / n_tiles`

At episode end: the step reward gets *overwritten* following one of these conditions:
* `reward = f` when track is finished, meaning all track tiles have been touched. Beware that driving on the grass does not touch the nearby tile!!!
* `reward = g` when patience is exceeded, meaning the game executed `patience * FPS`steps without touching a new tile
* `reward = h` when the car gets out of bounds
* `reward = i` when the episode exceeds 1000 steps (2000 for -v3) without finishing the track

For example, the original CarRacing-v0 and -v1 reward function looks like this:
`GYM_REWARD = [-0.1, 0.0, 0.0, 0.0, 10.0, 0,   0,  -100, 0]`

I set a slight modification as the std reward function for -v2:
`STD_REWARD = [-0.1, 0.0, 0.0, 0.0, 1.0, 100, -20, -100, -50]`

Beware neither of these are great selections, **you** should play around to understand the importance of reward functions!!


# Examples
Included 2 notebooks to get a quick overview of the methods needed to run and train this configurable environment.

The `Train_simple...` example just gets you rolling quickly, but might be a poor way of training a RL policy

The `Train_....+eval` example shows a more complete setup, including automatic evaluation callbacks, progressive model training and saving, and a nice final performance visualization
