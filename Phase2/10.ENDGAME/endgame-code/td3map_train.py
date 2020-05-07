# Self Driving Car
 
#region Importing the libraries

import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

import torch
import torchvision

import argparse
import os

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from PIL import ImageDraw
from kivy.graphics.texture import Texture

# Importing the TD3 model
import td3model
from td3model import TD3
from td3model import CNN
from td3model import ReplayBuffer

#endregion

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
goal_x = 1112
goal_y = 55

# TD3 constants

seed = 0 # Random seed number
start_timesteps = 1e3 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 1e4 # Total number of iterations/timesteps
save_models = True # Boolean checker whether or not to save the pre-trained model
expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
batch_size = 100 # Size of the batch
discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
tau = 0.005 # Target network update rate
policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

replay_buffer = ReplayBuffer()

state_dim = 20
action_dim = 1
max_action = 1.0

total_timesteps = 0
timesteps_since_eval = 0
episode_timesteps = 0
episode_num = 0
t0 = time.time()

policy = TD3(state_dim, action_dim, max_action)

last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

obs = None
new_img = None


# textureMask = CoreImage(source="./kivytest/simplemask1.png")


# Initializing the map
evaluations = []
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global obs

    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1420
    goal_y = 622
    first_update = False
    
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        


# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    cnn = None

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def obs(self):
        cropimg = PILImage.open("./images/MASK1.png")
        if (self.car.x > 40) and (self.car.y > 40):
            #print("**",self.car.x, self.car.y)
            box = (self.car.x - 40, self.car.y - 40, self.car.x + 40, self.car.y + 40)
            #print(orientation)
            crop = cropimg.crop(box)
            draw = ImageDraw.Draw(crop)
            draw.polygon([(35,35), (35, 45), (45, 35)], fill = 'yellow')
            return crop

    def getState(self, image):
        with torch.no_grad(): 
            self.cnn = CNN(3, 20)
            self.cnn_optim = torch.optim.Adam(self.cnn.parameters(), lr=0.001)
            image = image.convert("RGB")
            #state = torchvision.transforms.ToTensor()(image).unsqueeze(0).float()
            state = torch.FloatTensor((np.asarray(image).reshape(1, 3, 80, 80)))
            cnnoutput = self.cnn(state) 
            return (cnnoutput.cpu().detach().numpy()[0])

    def reset(self):
        self.car.x = 367
        self.car.y = 456

    # Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment

    def step(self, rotation):
        global last_reward
        #global scores
        global last_distance
        global goal_x
        global goal_y
        global done
        #global longueur
        #global largeur
        global swap
        
        self.car.move(rotation * 5)
        #self.car.move(0)
        #longueur = self.width
        #largeur = self.height
        #if first_update:
        #    init()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180
        
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            #print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.2
            print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.1
            else:
                last_reward = last_reward +(-0.2)

        if self.car.x < 5:
            self.car.x = 5
            last_reward = -1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -1
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -1
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -1

        if distance < 25:
            done = True
            if swap == 1:
                goal_x = 1112
                goal_y = 55
                swap = 0
            else:
                goal_x = 352
                goal_y = 349
                swap = 1
        else:   
            done = False

        last_distance = distance

    def evaluate_policy(self, policy, eval_episodes=10):
        global done

        avg_reward = 0.
        for _ in range(eval_episodes):
            self.reset()
            
            done = False
            while not done:
                action = policy.select_action(np.array(self.getState(self.obs())))
                self.step(action)
                obs = self.getState(self.obs())
                reward = last_reward
                avg_reward += reward
        avg_reward /= eval_episodes
        print ("---------------------------------------")
        print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
        print ("---------------------------------------")
        return avg_reward


    def update(self,dt):
        global done
        global longueur
        global largeur
        global last_reward
        global total_timesteps
        global timesteps_since_eval
        global episode_timesteps
        global episode_reward
        global max_timesteps
        global episode_num
        global t0
        global evaluations

        longueur = self.width
        largeur = self.height
        action = 0
        done = True

        if first_update:
            init()
            episode_reward = 0
            self.reset()
            obs = self.getState(self.obs())

        
        if total_timesteps < max_timesteps:
            # If the episode is done
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if total_timesteps % 1000 == 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy(policy))
                    policy.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations)
                
                # When the training step is done, we reset the state of the environment
                #self.reset()
                obs = self.getState(self.obs())
                
                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                #episode_reward = 0
                #episode_timesteps = 0
                episode_num += 1
            
            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = np.random.uniform(1,-1)
            else: # After 10000 timesteps, we switch to the model
                action = policy.select_action(np.array(self.getState(self.obs())))
                action = action[0].item()
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=1)).clip(-1, 1)
            
            # The agent performs the action in the environment, then reaches the next state and receives the reward
            self.step(action)
            new_obs = self.getState(self.obs())
            reward = last_reward
            # We check if the episode is done
            done_bool = 0 if episode_timesteps + 1 == 500 else float(done)
            
            # We increase the total reward
            episode_reward += reward
            
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, torch.tensor(action), reward, done_bool))
            #print((obs, new_obs, action, reward, done_bool))
            #print(len(replay_buffer.storage))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1

        else:   

            # We add the last policy evaluation to our list of evaluations and we save our model
            evaluations.append(self.evaluate_policy(policy))
            if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)


class CarApp(App):
    gameObj = None
    action = 0.0
    def build(self):
        parent = Game()
        gameObj = parent
        parent.serve_car()
        #parent.step(0)
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent
        

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
