"""
Easiest continuous control task to learn from pixels, a top-down racing environment.
Discrete control is reasonable in this environment as well, on/off discretization is fine.

State consists of STATE_W x STATE_H pixels.

The reward is -0.1 every frame and +100/N for every track tile visited, where N is 
the total number of tiles visited in the track. For example, if you have finished in 732 frames, 
your reward is 100 - 0.1*732 = 26.8 points.

The game is solved when the agent consistently gets 10+ points. 
Track generated can be random every episode, or maintained.

The episode finishes when all the tiles are visited. The car also can go outside of the PLAYFIELD,
that is far off the track, then it will get -100 and die.

Some indicators are shown at the bottom of the window and the state RGB buffer. 
From left to right: internal reward, true speed, steering wheel position and gyroscope.

To play yourself (it's rather fast for humans), type:

python gym/envs/box2d/car_racing_pab.py

Remember it's a powerful rear-wheel drive car -  don't press the accelerator and turn at the same time.

Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
Substantial modifications by Pablo Leslabay, posibly incompatible to original

"""

import math, os #,sys
import numpy as np

import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)

import gym
from gym import spaces
from gym.envs.box2d.car_dynamics import Car
from gym.utils import seeding, EzPickle #,colorize

import pyglet
from pyglet import gl
from PIL import Image


##SOME of this values are parameters that can be overwritten on __init__
# state frame and view/render size
STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 400
VIDEO_H = 400
WINDOW_W = 400
WINDOW_H = 400
ZOOM_START  = False       # Set to True for flying start zoom
TRACK_ZOOM  = 1           # Zoom for complete trackoverview
ZOOM        = 1.5         # Racing Camera zoom
TRACK_FIRST = True        # Set to True for whole track on first render
MAX_TIME_NEW_TILE = 1.0   # limits allowed time (n*FPS) without progress
MAX_TIME_GRASS = 2.0

COLOR  = 0                # 0 for RGB, 1 grayscale, 2 green channel
FPS    = 1/0.03           # Simulation Frames per second, timebase

### Discretization
## after investigating car_dynamics, is my understanding this model will really work BADLY in continous mode...
# physics model control actions, can be simultaneous => steering_angle, gas (throttle), brake_level
# due to car_dynamics setup, d(gas)/dt is limited to 0.1 per Frame, braking >=0.9 blocks wheels, only friction limited (1) currently
# due to car_dynamics, steering saturates @+-0.4; it takes 7 steps to fully turn wheels either side from center @ steering >= 0.4
# due to car_dynamics, the car presents lots of wheelspin. Gas < 1 might be a faster way to accelerate the car.

# continuos action = (steering, throttle, brake)
ACT = [[0, 0, 0], [-0.4, 0, 0], [0.4, 0, 0], [0, 1, 0], [0, 0, 0.8]]
# discrete actions: center_steering and no gas/brake, left, right, accel, brake  
#     --> actually a good choice, because car_dynamics softens the action's diff for gas and steering

## ACT2 is only useful for actually driving outside the track without drifting around
# center_steering and no gas/brake, left, right, accel, brake, half_left, half_right, half_accel, soft_brake   
ACT2= [[0, 0, 0], [-0.4, 0, 0], [0.4, 0, 0], [0, 0.8, 0], [0, 0, 0.8], [-0.2, 0, 0], [0.2, 0, 0], [0, 0.4, 0], [0, 0, 0.4]]

##REWARDS 
# reward given each step: step, distance to centerline, speed, steer angle
# reward given on new tile: % of advance
# reward given at episode end: finished, patience exceeded, out of bounds, steps exceeded
# reward for obstacles:  obstacle hit (each step), obstacle collided (episode end)
GYM_REWARD = [-0.1, 0.0, 0.0, 0.0,   10.0,     0,  -0, -100, -0,     -0, -0 ]
STD_REWARD = [-0.1, 0.0, 0.0, 0.0,    1.0,   100, -20, -100, -50,    -0, -0 ]
CONT_REWARD =[-0.1, 0.0, 0.0, 0.0,    1.0,   100, -20, -100, -50,    -5, -100 ]

# Pablo's style indicators
INDICATORS = True
H_INDI     = 4            # draw block heigth, total indicator bar 5 blocks

### Track generation, same algorithm as original, slightly different lenght. Should not be incompatible
# Try not to change these values, unexpected consequences may arise
SCALE       = 6.0          # Track scale
TRACK_DETAIL_STEP = 21/SCALE
TRACK_TURN_RATE = 0.31
BORDER      = 8/SCALE
BORDER_MIN_COUNT = 4
ROAD_COLOR  = [0.4, 0.4, 0.4]
GRASS_COLOR = [0.4 ,0.8 ,0.4]  #lighter grass patches uses [-0, +0.1, -0.]
BORDER_COLOR= (1,   0.15,0.0)  #and white [1,1,1]
OBSTACLE_COLOR= [0.2,0.2,0.9]
OILY_COLOR  = [0.2, 0.2, 0.2]
##color values taken from car_dynamics for reference, not to be changed here
#HULL_COLOR  = (0.8, 0  , 0)
#WHEEL_COLOR = (0.0, 0.0, 0.0)
#WHEEL_WHITE = (0.3, 0.3, 0.3)
#MUD_COLOR   = (0.4, 0.45,0.2) 

# here you can play
PLAYFIELD   = 1250/SCALE   # Game over boundary radius, at least 25% larger than TRACK_RAD
TRACK_RAD   = 1000/SCALE   # Track is heavily morphed circle with this radius
TRACK_COMPL = 12           # general geometrical complexity of the track, divides the circle in this much segments, to morph
TRACK_WIDTH = 40/SCALE     # proportional track width, in pixels


class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env
    def BeginContact(self, contact):
        self._contact(contact, True)
    def EndContact(self, contact):
        self._contact(contact, False)
    
    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj  = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj  = u1
        if not tile:
            return

        if tile.typename != 'obstacle':
            tile.color[0] = ROAD_COLOR[0]
            tile.color[1] = ROAD_COLOR[1]
            tile.color[2] = ROAD_COLOR[2]
        # else:
        #     tile.color[0] = 0.2
        #     tile.color[1] = 0.2
        #     tile.color[2] = 0.2
            
        if not obj or "tiles" not in obj.__dict__:
            return
        
        if begin:
            if tile.typename != 'obstacle':
                obj.tiles.add(tile)
                # print tile.road_friction, "ADD", len(obj.tiles)
                if not tile.road_visited:
                    tile.road_visited = True
                    if self.env.t > 2/FPS:
                        self.env.newtile = True                    
                    self.env.tile_visited_count += 1
                    self.env.last_new_tile = self.env.t
            else:
                print('obstacle hit')
                self.env.obst_contact = True
                self.env.obst_contact_count += 1
                self.env.obst_contact_list.append(tile.id)
        else:
            if tile.typename != 'obstacle':
                obj.tiles.remove(tile)
                # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)
                # Registering last contact with track
                self.env.last_touch_with_track = self.env.t
            else:
                self.env.obst_contact = False


class CarRacing2(gym.Env, EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array', 'state_pixels'],
        'FPS, 1/timebase': FPS, 
        #'discretization': DISCRETE, 
        'Zoom_level': ZOOM,
        'Flight start': ZOOM_START,
        'show_track_1st': TRACK_FIRST,
        'state_pixels frame size': [STATE_H, STATE_W],
        }

    def __init__(self, seed=None, **kwargs):
        EzPickle.__init__(self)
        self._seed(seed)
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0,0), contactListener=self.contactListener_keepref)
        self.viewer = None
        #self.invisible_state_window = None
        #self.invisible_video_window = None
        self.road = None
        self.car = None
        self.newtile = False
        self.ep_return = 0.0
        self.action_taken = +np.inf
        self.fd_tile = fixtureDef(
                shape = polygonShape(vertices=
                    [(0, 0),(1, 0),(1, -1),(0, -1)]))

        # Config
        self._set_config(**kwargs)
        #self._org_config = deepcopy(kwargs)
    
    def _set_config(self, 
            use_track = 1,                  # number of times to use the same Track, [1-100]. More than 20 high risk of overfitting!!
            episodes_per_track = 1,         # number of evenly distributed Starting Points on each track [1-20]
            discre = ACT,                   # Action discretization function, format [[steer0, throtle0, brake0], [steer1, ...], ...]. None for continous
            tr_complexity = TRACK_COMPL,    # generated Track geometric Complexity, [6-20]
            tr_width = TRACK_WIDTH,         # relative Track Width, [30-50]
            patience = MAX_TIME_NEW_TILE,   # max time in secs without Progress, [0.5-20]
            off_track = MAX_TIME_GRASS,     # max time in secs Driving on Grass, [0-5]
            indicators = True,              # show or not bottom info Panel
            game_color = 1,                 # State color option: 0 = RGB, 1 = Grayscale, 2 = Green only
            frames_per_state = 1,           # stacked (history) Frames on each state [1-inf]
            skip_frames = 0,                # number of Frames to skip on history, latest observation always on first Frame [0-4]
            f_reward = CONT_REWARD,         # Reward Funtion coeficients, refer to Docu for details
            num_obstacles = 5,              # Obstacle objects randomly placed on track [0-10]
            end_on_contact = False,         # stop episode on contact with obstacle, not recommended for starting-phase of training
            obst_location = 0,              # array pre-setting obstacle Location, in %track. Negative value means tracks's left-hand side. 0 for random location
            oily_patch = False,             # use obstacle as Low-friction road (oily patch)
            verbose = 2      ):
        
        #Verbosity
        self.verbose = verbose

        #obstacles        
        self.num_obstacles = np.clip(num_obstacles, 0, 10)
        self.end_on_contact = end_on_contact        
        self.oily_patch = oily_patch
        if obst_location != 0 and len(obst_location) < num_obstacles:
            print("#####################################")
            print("Warning: incomplete obstacle location")
            print("Defaulting to random placement")
            self.obst_location = 0 #None
        else:
            self.obst_location = np.array(obst_location)
        
        #reward coefs verification
        if len(f_reward) < len(CONT_REWARD):
            print("####################################")
            print("Warning: incomplete reward function")
            print("Defaulting to predefined function!!!")
            self.f_reward = CONT_REWARD
        else:
            self.f_reward = f_reward

        # Times to use same track, up to 100 times. More than 20 high risk of overfitting!!
        self.repeat_track = np.clip(use_track, 1, 100)
        self.track_use = +np.inf

        # Number of episodes on same track, with evenly distributed starting points, 
        # not more than 20 episodes
        self.episodes_per_track = np.clip(episodes_per_track, 1, 20)

        # track generation complexity
        self.complexity = np.clip(tr_complexity, 6, 20)
        
        # track width
        self.tr_width = np.clip(tr_width, 30, 50)/SCALE

        # Max time without progress
        self.patience = np.clip(patience, 0.5, 20)
        # Max time off-track
        self.off_track = np.clip(off_track, 0, 5)

        # Show or not bottom info panel
        self.indicators = indicators

        # Grayscale and acceptable frames
        self.grayscale = game_color
        if not self.grayscale: 
            if frames_per_state > 1:
                print("####################################")
                print("Warning: making frames_per_state = 1")
                print("No support for several frames in RGB")
                frames_per_state = 1
                skip_frames = 0

        # Frames to be skipped from state (max 4)
        self.skip_frames = np.clip(skip_frames+1, 1, 5)

        # Frames per state
        self.frames_per_state = frames_per_state if frames_per_state > 0 else 1
        if self.frames_per_state > 1:
            lst = list(range(0, self.frames_per_state*self.skip_frames, self.skip_frames))
            self._update_index = [lst[-1]] + lst[:-1]
        
        # Gym spaces, observation and action    
        self.discre = discre
        if discre==None:
            self.action_space = spaces.Box(np.array([-0.4,0,0]), np.array([+0.4,+1,+1]), dtype=np.float32)  # steer, gas, brake
        else:
            self.action_space = spaces.Discrete(len(discre)) 
        
        if game_color:
            self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, self.frames_per_state), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)


    def _update_state(self, new_frame):
        if self.frames_per_state > 1:
            self.int_state[:,:,-1] = new_frame
            self.state = self.int_state[:,:,self._update_index]
            self.int_state = np.roll(self.int_state,1,2)
        else:
            self.state = new_frame

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _create_track(self):
        # Create checkpoints
        CHECKPOINTS = self.complexity
        checkpoints = []
        for c in range(CHECKPOINTS):
            alpha = 2*math.pi*c/CHECKPOINTS + self.np_random.uniform(0, 2*math.pi*1/CHECKPOINTS)
            rad = self.np_random.uniform(TRACK_RAD/3, TRACK_RAD)
            if c==0:
                alpha = 0
                rad = 1.0*TRACK_RAD
            if c==CHECKPOINTS-1:
                alpha = 2*math.pi*c/CHECKPOINTS
                self.start_alpha = 2*math.pi*(-0.5)/CHECKPOINTS
                rad = 1.0*TRACK_RAD
            checkpoints.append( (alpha, rad*math.cos(alpha), rad*math.sin(alpha)) )

        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        
        # Go from one checkpoint to another to create track
        x, y, beta = 1.0*TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        waypoint = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2*math.pi
            while True: # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2*math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x*dest_dx + r1y*dest_dy  # destination vector projected on rad
            while beta - alpha >  1.5*math.pi:
                 beta -= 2*math.pi
            while beta - alpha < -1.5*math.pi:
                 beta += 2*math.pi
            prev_beta = beta
            proj *= SCALE
            if proj >  0.3:
                 beta -= min(TRACK_TURN_RATE, abs(0.001*proj))
            if proj < -0.3:
                 beta += min(TRACK_TURN_RATE, abs(0.001*proj))
            x += p1x*TRACK_DETAIL_STEP
            y += p1y*TRACK_DETAIL_STEP
            track.append( (alpha, prev_beta*0.5 + beta*0.5, x, y) )
            waypoint.append([x, y])
            if laps > 4:
                 break
            no_freeze -= 1
            if no_freeze==0:
                 break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i==0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha and track[i-1][0] <= self.start_alpha
            if pass_through_start and i2==-1:
                i2 = i
            elif pass_through_start and i1==-1:
                i1 = i
                break
        if self.verbose > 0:
            print("Track generation: %i..%i -> %i-tiles track, complex %i" % (i1, i2, i2-i1, self.complexity))
        assert i1!=-1
        assert i2!=-1

        track = track[i1:i2-1]
        waypoint = waypoint[i1:i2-1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square( first_perp_x*(track[0][2] - track[-1][2]) ) +
            np.square( first_perp_y*(track[0][3] - track[-1][3]) ))
        if well_glued_together > TRACK_DETAIL_STEP:
            return False   # Failed

        # Red-white border on hard turns, pure colors
        border = [False]*len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i-neg-0][1]
                beta2 = track[i-neg-1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE*0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i-neg] |= border[i]
        
        # Get random tile for obstacles, without replacement
        if np.sum(self.obst_location) == 0:
            obstacle_tiles_ids = np.random.choice(range(10, len(track)-6), self.num_obstacles, replace=False)
            obstacle_tiles_ids *= (np.random.randint(0,2,self.num_obstacles)*2-1)
            #obstacle_tiles_ids[0] = 4
        else:
            obstacle_tiles_ids = np.rint(self.obst_location*len(track)/100).astype(int)
            obstacle_tiles_ids = obstacle_tiles_ids[0:self.num_obstacles]
        if self.verbose >= 2:
            print(self.num_obstacles, ' obstacles on tiles: ', obstacle_tiles_ids[np.argsort(np.abs(obstacle_tiles_ids))] )

        #stores values and call tile generation
        self.border = border
        self.track = track
        self.waypoints = np.asarray(waypoint)
        self.obstacle_tiles_ids = obstacle_tiles_ids
        self._create_tiles(track, border)
        
        return True  #self.waypoint #True

    def _give_track(self):
        return self.track, self.waypoints, self.obstacles_poly

    def _create_tiles(self, track, border):
        # first you need to clear everything
        if self.road is not None:
            for t in self.road:
                self.world.DestroyBody(t)
        self.road = []
        self.road_poly = []

        # Create track tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i-1]
            road1_l = (x1 - self.tr_width*math.cos(beta1), y1 - self.tr_width*math.sin(beta1))
            road1_r = (x1 + self.tr_width*math.cos(beta1), y1 + self.tr_width*math.sin(beta1))
            road2_l = (x2 - self.tr_width*math.cos(beta2), y2 - self.tr_width*math.sin(beta2))
            road2_r = (x2 + self.tr_width*math.cos(beta2), y2 + self.tr_width*math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.02*(i%3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.typename = 'tile'
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(( [road1_l, road1_r, road2_r, road2_l], t.color ))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side* self.tr_width        *math.cos(beta1), y1 + side* self.tr_width        *math.sin(beta1))
                b1_r = (x1 + side*(self.tr_width+BORDER)*math.cos(beta1), y1 + side*(self.tr_width+BORDER)*math.sin(beta1))
                b2_l = (x2 + side* self.tr_width        *math.cos(beta2), y2 + side* self.tr_width        *math.sin(beta2))
                b2_r = (x2 + side*(self.tr_width+BORDER)*math.cos(beta2), y2 + side*(self.tr_width+BORDER)*math.sin(beta2))
                self.road_poly.append(( [b1_l, b1_r, b2_r, b2_l], (1,1,1) if i%2==0 else BORDER_COLOR ))
        
        #create obstacles tiles
        if self.num_obstacles:
            self._create_obstacles()

    def _create_obstacles(self):
        # Create obstacle (blue rectangle of fixed width and randomish position in tile)
        count=1
        self.obstacles_poly = []
        width = self.tr_width/2
        obst_len = 3 if self.oily_patch else 1
        for idx in self.obstacle_tiles_ids:
            if idx < 0:
                idx = -idx
                alpha1, beta1, x1, y1 = self.track[idx]
                alpha2, beta2, x2, y2 = self.track[idx+obst_len]
                p1 = (x1 - width*math.cos(beta1), y1 - width*math.sin(beta1))
                p2 = (x1, y1)
                p3 = (x2, y2)
                p4 = (x2 - width*math.cos(beta2), y2 - width*math.sin(beta2))
            else:
                alpha1, beta1, x1, y1 = self.track[idx]
                alpha2, beta2, x2, y2 = self.track[idx+obst_len]
                p1 = (x1, y1)
                p2 = (x1 + width*math.cos(beta1), y1 + width*math.sin(beta1))
                p3 = (x2 + width*math.cos(beta2), y2 + width*math.sin(beta2))
                p4 = (x2, y2)

            vertices = [p1,p2,p3,p4]
            
            # Add it to obstacles, Add it to poly_obstacles
            t = self.world.CreateStaticBody(fixtures=fixtureDef(shape=polygonShape(vertices=vertices)))
            t.userData = t
            if self.oily_patch:
                t.color = OILY_COLOR 
                t.road_friction = 0.2
            else:
                t.color = OBSTACLE_COLOR 
                t.road_friction = 1.0
            t.typename = 'obstacle'
            t.road_visited = False
            t.id = count
            t.tile_id = idx
            t.fixtures[0].sensor = True
            self.road.append(t)
            self.obstacles_poly.append(( vertices, t.color ))
            count += 1
    
    def _closest_node(self, node, nodes):
        #nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.argmin(dist_2)

    def _closest_dist(self, node, nodes):
        #nodes = np.asarray(nodes)
        deltas = nodes - node
        dist_2 = np.einsum('ij,ij->i', deltas, deltas)
        return np.sqrt(min(dist_2))

    def _render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(GRASS_COLOR[0], GRASS_COLOR[1], GRASS_COLOR[2], 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        
        gl.glColor4f(GRASS_COLOR[0]-0, GRASS_COLOR[1]+0.1, GRASS_COLOR[2]-0, 1.0)
        k = PLAYFIELD/20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k*x + k, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + 0, 0)
                gl.glVertex3f(k*x + 0, k*y + k, 0)
                gl.glVertex3f(k*x + k, k*y + k, 0)
        
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        
        if self.num_obstacles > 0:
            self._render_obstacles()
        
        gl.glEnd()

    def _render_obstacles(self):
        #Can only be called inside a glBegin!!!
        for poly, color in self.obstacles_poly:    # drawing road old way
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)

    def _render_indicators(self, W, H):
        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h + h*val, 0)
            gl.glVertex3f((place+1)*s, h, 0)
            gl.glVertex3f((place+0)*s, h, 0)
        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place+0)*s, 4*h , 0)
            gl.glVertex3f((place+val)*s, 4*h, 0)
            gl.glVertex3f((place+val)*s, 2*h, 0)
            gl.glVertex3f((place+0)*s, 2*h, 0)

        s = W/4 #horizontal slot separation
        #h = H_INDI   #vertical pixels definition
        h = H / 40.0
        
        #black bar, 5x h height
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0,0,0,1)
        gl.glVertex3f(W, 0, 0)
        gl.glVertex3f(W, 5*h, 0)
        gl.glVertex3f(0, 5*h, 0)
        gl.glVertex3f(0, 0, 0)
        
        #3 hor indicators
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        #vertical_ind(5, 0.02*true_speed, (1,1,1))
        horiz_ind(1.0, 0.015*true_speed, (1,1,1))
        horiz_ind(2.5, -1*self.car.wheels[0].joint.angle, (0,1,0))
        horiz_ind(3.5, np.clip(-0.03*self.car.hull.angularVelocity,-0.4,0.4), (1,1,0))
        #vertical_ind(7, 0.01*self.car.wheels[0].omega, (0.0,0,1)) # ABS sensors
        #vertical_ind(8, 0.01*self.car.wheels[1].omega, (0.0,0,1))
        #vertical_ind(9, 0.01*self.car.wheels[2].omega, (0.2,0,1))
        #vertical_ind(10,0.01*self.car.wheels[3].omega, (0.2,0,1))
        gl.glEnd()
        
        #total_reward
        self.score_label.text = "%02.1f" % self.ep_return
        self.score_label.draw()

    def reset(self):
        self.ep_return = 0.0
        self.newtile = False
        self.tile_visited_count = 0
        self.last_touch_with_track = 0
        self.last_new_tile = 0
        self.obst_contact = False
        self.obst_contact_count = 0
        self.obst_contact_list=[]
        self.t = 0.0
        self.steps_in_episode = 0
        self.state = np.zeros(self.observation_space.shape)
        self.internal_frames = self.skip_frames*(self.frames_per_state-1) +1
        self.int_state = np.zeros([STATE_H, STATE_W, self.internal_frames])
        
        if self.track_use >= self.repeat_track*self.episodes_per_track: 
            intento=0
            while intento < 21:
                success = self._create_track()
                intento += 1
                if success:
                    self.track_use = 0
                    self.episode_start = range(0, len(self.track), int(len(self.track)/self.episodes_per_track))                        
                    #print(self.episode_start)
                    break
                if self.verbose > 0:
                    print(intento," retry to generate new track (normal below 10, limit 20)")
        else:
            self._create_tiles(self.track, self.border)

        start_tile = self.episode_start[self.track_use % self.episodes_per_track]
        #print(start_tile, self.track_use, self.episodes_per_track)
       
        if self.car is not None:
            self.car.destroy()
        if self.episodes_per_track > 1:
            self.car = Car(self.world, *self.track[start_tile][1:4])
        else:
            self.car = Car(self.world, *self.track[0][1:4])

        #trying to detect two very close reset()        
        if self.action_taken > 2:
            self.track_use += 1
            self.action_taken = 0
        #self.track_use += 1
 
        return self.step(None)[0]

    def reset_track(self):
        self.track_use = +np.inf
        self.reset()
        return self.step(None)[0]

    def step(self, action):
        # Avoid first step with action=None, called from reset()
        if action is None:   
            #render car and environment
            self.car.steer(0)
            self.car.step(0)
            self.world.Step(0, 6*30, 2*30)
            #step_reward = 0
            #self.state(self.render("state_pixels"))
        else:
            if not self.discre==None:
                action = self.discre[action]
            #moves the car per action, advances time    
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])
            self.t += 1.0/FPS
            self.steps_in_episode += 1
            self.action_taken += 1
            #render car and environment
            self.car.step(1.0/FPS)
            self.world.Step(1.0/FPS, 6*30, 2*30)
            
        #generates new observation state
        #self.state[:,:,0] = self.render("state_pixels") # Old code, only one frame
        self._update_state(self.render("state_pixels"))

    ##REWARDS 
        # reward given each step: step, distance to centerline, speed, steer angle
        # reward given on new tile: % of advance
        # reward given at episode end: finished, patience exceeded, out of bounds, steps exceeded
        # reward for obstacles:  obstacle hit (each step), obstacle collided (episode end)
        x, y = self.car.hull.position
        true_speed = np.sqrt(np.square(self.car.hull.linearVelocity[0]) + np.square(self.car.hull.linearVelocity[1]))
        done = False

        #reward for each step taken       
        step_reward = self.f_reward[0]

        #reward distance to centerline, proportional to trackwidth
        dist = 1 - self._closest_dist([x,y], self.waypoints)/self.tr_width
        step_reward += self.f_reward[1]*np.clip(dist, -1, 1)

        #reward for speed
        step_reward += self.f_reward[2]*true_speed

        #reward for steer angle
        step_reward += self.f_reward[3]*abs(self.car.wheels[0].joint.angle)
        
        #reward for collision with obstacle
        step_reward += self.f_reward[9]*self.obst_contact

        #reward new tile touched
        if self.newtile:        
            step_reward += self.f_reward[4]*100/len(self.track)
            self.newtile = False

        ## calculates reward penalties, showstopper
        # check collision with obstacle
        if self.end_on_contact and self.obst_contact:
            step_reward = self.f_reward[10]
            done = True
            if self.verbose > 0:
                print(self.track_use," ended by collision. Steps", self.steps_in_episode, 
                      " %advance", int(self.tile_visited_count/len(self.track)*1000)/10,
                      " played reward", int(100*self.ep_return)/100, " last penalty", step_reward)
            if self.verbose > 2:
                print(self.obst_contact_count, "  collided obstacles: ", self.obst_contact_list) 

        # if too many seconds lacking progress
        if self.t - self.last_new_tile > self.patience:
            step_reward = self.f_reward[6]
            done = True
            if self.verbose > 0:
                print(self.track_use," cut by time without progress. Steps", self.steps_in_episode, 
                      " %advance", int(self.tile_visited_count/len(self.track)*1000)/10,
                      " played reward", int(100*self.ep_return)/100, " last penalty", step_reward)
        
        # if too many seconds off-track
        if self.t - self.last_touch_with_track > self.off_track:
            step_reward = self.f_reward[6]
            done = True
            if self.verbose > 0:
                print(self.track_use," cut by time off-track. Steps", self.steps_in_episode, 
                      " %advance", int(self.tile_visited_count/len(self.track)*1000)/10,
                      " played reward", int(100*self.ep_return)/100, " last penalty", step_reward)

        #check out-of-bounds car position
        if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
            step_reward = self.f_reward[7]
            done = True
            if self.verbose > 0:
                print(self.track_use," out of limits. Steps", self.steps_in_episode, 
                      " %advance", int(self.tile_visited_count/len(self.track)*1000)/10,
                      " played reward", int(100*self.ep_return)/100, " last penalty", step_reward)

        #episode limit, as registered
        if self.steps_in_episode >= 2000:
            step_reward = self.f_reward[8]
            done = True
            if self.verbose > 0:
                print(self.track_use, " env max steps reached", self.steps_in_episode, 
                      " %advance", int(self.tile_visited_count/len(self.track)*1000)/10,
                      " played reward", int(100*self.ep_return)/100, " last penalty", step_reward)

        # check touched all tiles, to finish
        if self.tile_visited_count==len(self.track):
            step_reward = self.f_reward[5]
            done = True
            if self.verbose > 0:
                print(self.track_use, " Finalized in Steps", self.steps_in_episode, 
                      " with return=total_reward", self.ep_return+step_reward)

        #clear reward if no action intended, from reset
        if action is None: step_reward = 0
            
        #internal counting reward, for display
        self.ep_return += step_reward
              
        return self.state, step_reward, done, {}   #{'episode', self.tile_visited_count/len(self.track)} 

    def render(self, mode='human'):
        assert mode in ['human', 'state_pixels', 'rgb_array']
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label('00.0', font_size=24,
                x=10, y=WINDOW_H * 2.5 / 40.00,  #2.5*H_INDI, 
                anchor_x='left', anchor_y='center',
                color=(255,255,255,255))
            self.transform = rendering.Transform()

        if "t" not in self.__dict__: return  # reset() not called yet
        
        if ZOOM_START:   # Animate zoom during first second
            zoom = 0.1*SCALE*max(1-self.t, 0) + ZOOM*SCALE*min(self.t, 1)   
        else:
            zoom = ZOOM*SCALE

        if TRACK_FIRST and self.t == 0:  #shows whole track in first frame; checks first step, from reset()
            self.transform.set_scale(TRACK_ZOOM, TRACK_ZOOM)
            self.transform.set_translation(WINDOW_W/2, WINDOW_H/2)
            self.transform.set_rotation(0)
        else:           #every regular step updates the car visualization after action
            scroll_x = self.car.hull.position[0]
            scroll_y = self.car.hull.position[1]
            angle = -self.car.hull.angle
            vel = self.car.hull.linearVelocity
            if np.linalg.norm(vel) > 0.5:
                angle = math.atan2(vel[0], vel[1])
            self.transform.set_scale(zoom, zoom)
            self.transform.set_translation(
                    WINDOW_W/2 - (scroll_x*zoom*math.cos(angle) - scroll_y*zoom*math.sin(angle)),
                    WINDOW_H/4 - (scroll_x*zoom*math.sin(angle) + scroll_y*zoom*math.cos(angle)) )
            self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode!="state_pixels")  
        #car_dynamics.draw particles only when not in state_pixels

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == 'rgb_array':
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == 'state_pixels':
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, '_nscontext'):
                pixel_scale = win.context._nscontext.view().backingScaleFactor()  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self._render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        
        # plots the indicators
        if self.indicators and (not TRACK_FIRST or self.t >= 1.0/FPS):
#            self._render_indicators(VP_W, VP_H)
            self._render_indicators(WINDOW_W, WINDOW_H)

        if mode == 'human':
            win.flip()
            return self.viewer.isopen

        image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep='')
        arr = arr.reshape(VP_H, VP_W, 4)
        
        if self.grayscale==1:
            if self.frames_per_state >1:
                arr = np.dot(arr[::-1, :, 0:3], [0.299, 0.587, 0.114])
            else:
                arr = np.dot(arr[::-1, :, 0:3], [0.299, 0.587, 0.114]).reshape(VP_H, VP_W, -1)
        elif self.grayscale==2:
            #arr = np.expand_dims(arr[:,:,1], axis=-1, dtype=np.uint8)
            if self.frames_per_state >1:
                arr = arr[::-1, :, 1]
            else:
                arr = arr[::-1, :, 1].reshape(VP_H, VP_W, -1)
        else:
            arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def screenshot(self, dest="./", name=None,quality='low'):
        ''' 
        Saves the current state, quality 'low','medium' or 'high', low will save the 
        current state if the quality is low, otherwise will save the current frame
        '''
        if quality == 'low':
            state = self.state
        elif quality == 'medium':
            state = self.render('rgb_array')
        else:
            state = self.render("HD")
        if state is not None:
            for f in range(self.frames_per_state):

                if self.frames_per_state == 1 or quality != 'low':
                    frame_str = ""
                    frame = state
                else:
                    frame_str = "_frame%i" % f
                    frame = state[:,:,f]

                if self.grayscale:
                    frame = np.stack([frame,frame,frame], axis=-1)

                frame = frame.astype(np.uint8)
                im = Image.fromarray(frame)
                if name == None: name = "screenshot_%0.3f" % self.t
                im.save("%s/%s%s.jpeg" % (dest, name, frame_str))


if __name__=="__main__":
    from pyglet.window import key
    #a = np.array( [0.0, 0.0, 0.0] )
    aa = 0
    MAX_TIME_NEW_TILE= 20.0     # limits allowed time (n*FPS) without progress

    def key_press(k, mod):
        global restart
        if k==0xff0d: restart = True
        if k==key.LEFT:  aa=1  #a[0] = -1.0
        if k==key.RIGHT: aa=2  #a[0] = +1.0
        if k==key.UP:    aa=3  #a[1] = +1.0
        if k==key.DOWN:  aa=4  #a[2] = +0.8   # set 1.0 for wheels to block to zero rotation
    def key_release(k, mod):
        if k==key.LEFT:  aa=0  #and a[0]==-1.0: a[0] = 0
        if k==key.RIGHT: aa=0  #and a[0]==+1.0: a[0] = 0
        if k==key.UP:    aa=0  #a[1] = 0
        if k==key.DOWN:  aa=0  #a[2] = 0
    
    env = CarRacing2()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor
        env = Monitor(env, '/tmp/video-test', force=True)
    isopen = True
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            s, r, done, info = env.step(aa)
            total_reward += r
            if steps % 100 == 0 or done:
                #print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
                print("step {} total_reward {:+0.2f}".format(steps, total_reward))
                #import matplotlib.pyplot as plt
                #plt.imshow(s)
                #plt.savefig("test.jpeg")
            steps += 1
            isopen = env.render()
            if done or restart or isopen == False:
                break
    env.close()
