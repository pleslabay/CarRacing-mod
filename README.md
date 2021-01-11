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

* CarRacing-v3 maintains all of -v2 improvements, and adds the option to include *objects on track*. 
These objects can be used -in combination with a proper reward function!- to train obstacle avoidance behavior. 
It is also possible to define if the object should act as a low-friction pad, to induce car-spinning.
The implementation allows for automatically, randomly placed objects, or for a preset location on track. 

(*) **Note:** From *gym version 0.17.3* onward, openAI provides a pyglet optimized environment, running up to 50% faster, as tested on my hardware. Although, I am experimenting memory leak problems with that environment, so I decided to roll back to the original (slower) version for now.

# Installation

It is highly recommendable that you start your experimentation with a new environment, virtual or conda managed. Python 3.6 will do well for now.
Then `pip install gym`, following when on Linux distro with `pip install 'gym[box2d]'  #all dependencies for CarRacing` or when on Win10 (preferably with WSL2 enabled!) execute `conda install -c conda-forge box2d-py ` from a terminal, with the right environment activated.

As my Reinforcement Learning toolkit I have been using *Stable baselines 2*. Its a great project with a very friendly documentation. Beware of TensorFlow version limitations with this toolkit!
```
pip install stable-baselines
pip install tensorflow==1.15.0
```
Feel free to use any RL tools you like, just make sure to understand the mechanics behind the *gym environments*

Once finished installing, you will need to manually copy/overwrite the provided files, even when on a Colab instance. 
Sorry for this, but is way faster than any wheel implementation...
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

The original envs CarRacing-v0 and -v1 work with a physics core at 50 frames/sec (FPS), but does not need to be real-time. Meaning that you would really like your hardware to produce more than 50fps when training!
In order to decrease the training time, I experimented with *car_dynamics* and could get it to work properly @33FPS in CarRacing-v2 and -v3.
So be aware that a trained agent will probably take wrong decisions when crossing environments.

CarRacing-v2 and -v3 provide a set of customization parameters, as follows:
```
env = gym.make('CarRacing-v2',
      game_color = 0,           # State (frame) color option: 0 = RGB, 1 = Grayscale, 2 = Green only
      indicators = True,        # show or not bottom Info Panel
      frames_per_state = 1,     # stacked (rolling history) Frames on each state [1-inf], latest observation always on first Frame
      skip_frames = 1,          # number of consecutive Frames to skip between state saves [0-4]
      discre = ACT,             # Action discretization function, format [[steer0, throtle0, brake0], [steer1, ...], ...]. None for continuous
      
      use_track = 1,            # number of times to use the same Track, [1-100]. More than 20 high risk of overfitting!!
      episodes_per_track = 1,   # number of evenly distributed starting points on each track [1-20]. Every time you call reset(), the env automatically starts at the next point
      tr_complexity = 12,       # generated Track geometric Complexity, [6-20]
      tr_width = 40,            # relative Track Width, [30-50]
      patience = 2.0,           # max time in secs without Progress, [0.5-20]
      off_track = 1.0,          # max time in secs Driving on Grass, [0.0-5]
      f_reward = STD_REWARD,    # Reward Funtion coefficients, refer to Docu for details
      verbose = 1      )


env = gym.make('CarRacing-v3',
      game_color = 1,           # State (frame) color option: 0 = RGB, 1 = Grayscale, 2 = Green only
      indicators = True,        # show or not bottom Info Panel
      frames_per_state = 4,     # stacked (rolling history) Frames on each state [1-inf], latest observation always on first Frame
      skip_frames = 3,          # number of consecutive Frames to skip between state saves [0-4]
      discre = ACT*,            # Action discretization function, format [[steer0, throtle0, brake0], [steer1, ...], ...]. None for continuous
      
      use_track = 3,            # number of times to use the same Track, [1-100]. More than 20 high risk of overfitting!!
      episodes_per_track = 5,   # number of evenly distributed starting points on each track [1-20]. Every time you call reset(), the env automatically starts at the next point
      tr_complexity = 12,       # generated Track geometric Complexity, [6-20]
      tr_width = 45,            # relative Track Width, [30-50]
      patience = 2.0,           # max time in secs without Progress, [0.5-20]
      off_track = 1.0,          # max time in secs Driving on Grass, [0.0-5]
      f_reward = CONT_REWARD,   # Reward Funtion coefficients, refer to Docu for details
      num_obstacles = 5,        # Obstacle objects placed on track [0-10]
      end_on_contact = False,   # Stop Episode on contact with obstacle, not recommended for starting-phase of training
      obst_location = 0,        # array pre-setting obstacle Location, in %track. Negative value means tracks's left-hand side. 0 for random location
      oily_patch = False,       # use all obstacles as Low-friction road (oily patch)
      verbose = 2      )

```
All the shown values represent the default configuration, making -v2 almost equal to -v1. 
Being -v3 a completely different environment, the default presets try to favor a simpler training.

It makes sense to change and play around with these parameters!
The user can replace ACT and STD_REWARD/CONT_REWARD with custom np.arrays, in order to adapt *discre* and *f_reward* to his/her needs.
Remember you can set the obstacles placement passing a np.array to *obst_location*

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
The original game standard continuous action is a tuple  `(steering, throttle, brake)`
with designated action space `(-1 to 1 ; 0 to 1 ; 0 to 1)`. Please refer to the comments above regarding the sanity of those chosen limits.

So for starters, the simplest option to get around this difficult initial setting would be to discretize as follows:
`ACT = [[0, 0, 0], [-0.4, 0, 0], [0.4, 0, 0], [0, 1, 0], [0, 0, 0.8]]`
discrete actions: center_steering and no gas/brake, steer left, steer right, accel, brake  
     --> actually a good choice, because *car_dynamics* softens the action's diff for gas and steering

ACT2 is only useful for actually driving outside the track without drifting around, as it provides intermediate levels, thus making the training harder.
`ACT2= [[0, 0, 0], [-0.4, 0, 0], [0.4, 0, 0], [0, 0.8, 0], [0, 0, 0.8], [-0.2, 0, 0], [0.2, 0, 0], [0, 0.4, 0], [0, 0, 0.4]]`
discrete actions: center_steering and no gas/brake, steer left, steer right, accel, brake, half_left, half_right, half_accel, soft_brake

**Note**: For driving and avoiding obstacles, initial tests have shown that using less throttle is very beneficial. So for `-v3` the default discretization is slightly different:
`ACT* = [[0, 0, 0], [-0.4, 0, 0], [0.4, 0, 0], [0, 0.6, 0], [0, 0, 0.8]]`


### Obstacles
For -v3 you can add up to *10 obstacles on track*, and program cars behavior when encountering them via the reward function.
By default, these obstacles are located randomly on track, mixed either at the left- or the right-hand side of the track, thus requiring the agent to learn to pass them on the opposite side. (Actually the car also fits on the smaller gap, and has been spotted learning that behavior on tests)

For your convenience, you can also pass a np.array with *preset obstacle location*, to aid the initial stages of training. 
For example, I use this for my initial training, then switch to random location by passing a 0.
`obst_loc = [6, -12, 25, -50, 75, -37, 62, -87, 95, -29]  #track percentage, negative for obstacle to the left-hand side`


### REWARDS 
For this environment I propose a fixed, parameterized reward function --> no scripting required
By selecting 10 or 12 parameters you can change the reward function behavior, as follows:

`SR = [a, b, c, d,	e, f,	 g, h, i, j,   k, m]`

After each step: 
`reward = a + b * dist_center + c * car_vel_mag + d * abs(car_steer_angle)` 
representing penalties for: 
* each step taken 
* distance to track's centerline, `dist_center = (1 - dist_to_nearest_waypoint / track_width)`, clipped to +-1 and relative to track width
* current car linear speed magnitude, normalized to [0-1]. Note that any well trained agent will rarely go faster than 0.75! 
* current car wheel steer angle (not necessary *your* steering input!), normalized to [0-1]

Each time a new tile is reached: 
`reward = reward  +  e * 100 / n_tiles  +  f * visited_tiles / steps_taken`

At episode end: the step reward gets *overwritten* following one of these conditions:
* `reward = g` when track is finished, meaning all track tiles have been touched. Beware that driving on the grass does not touch the nearby tile!!!
* `reward = h` when patience is exceeded, meaning the game executed `patience * FPS`steps without touching a tile that has not been visited before
* `reward = h` when driving off-track is exceeded, meaning the game executed `off_track * FPS`steps without touching any track tile
* `reward = i` when the car gets out of play-bounds
* `reward = j` when the episode exceeds 1000 steps (2000 for -v3) without finishing the track


#### For simulations with obstacles (-v3), the programed behavior goes as follows:
After each step: 
`reward = parameters from -v2  +  k * obstacle_is_being_hit` 

Each time a new tile is reached: 
`reward = reward  +  e * 100 / n_tiles  +  f * visited_tiles / steps_taken`

At episode end: the step reward gets *overwritten* as in -v2, or by this new condition:
* `reward = m` when the car has hit *any object* during the episode at least one time


As an example, the original CarRacing-v0 and -v1 reward function looks like this:
`GYM_REWARD = [-0.1, 0.0, 0.0, 0.0,  10.0, 0.0,    0,   0,  -100, 0]`

This slight modification serves as the default reward function for -v2, but following gym spirit:
`STD_REWARD = [-0.1, 0.0, 0.0, 0.0,   1.0, 0.0,   100, -20, -100, -50]`

For -v3 you need to add 2 additional parameters for the *obstacle hitting* behavior to the default
`CONT_REWARD =[-0.11, 0.1, 0.0, 0.0,  1.0, 0.0,   100, -20, -100, -50,    -5, -100 ]`

Beware neither of these are great selections, **you** should play around with the parameters to understand the importance of Reward Functions!!
*Note: you can pass a np.array with 12 parameters to either environment, only those needed will be used. Failing to provide enough parameters will automatically fall-back to the default values.


# Examples
Included 2 notebooks to get a quick overview of the methods needed to run and train this configurable environment.

The `Train_simple...` example just gets you rolling quickly, but might be a poor way of training a RL policy

The `Train_....+eval` example shows a more complete setup, including automatic evaluation callbacks, progressive model training and saving, and a nice final performance visualization

There is a different version of each of these notebooks for *tracks with obstacles.*

Currently working on a paralellizable and Colab suitable example!