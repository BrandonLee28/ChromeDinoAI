import optuna
from mss import mss
import pyautogui
import cv2
import numpy as np
import pytesseract
from matplotlib import pyplot as plt
import time
import gymnasium
from gymnasium import spaces
from stable_baselines3 import DQN
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import env_checker
   

class WebGame(gymnasium.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(low=0,high=255,shape=(1,100,280), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)
        self.cap = mss()
        self.game_location = {'top':200,'left':3225,'width':280,'height':100}
        self.done_location = {'top':192,'left':3415,'width':220,'height':20}
        self.seed = None

        

    def step(self,action):
        action_map = {0:'space',1:'down',2:'noop'}
        if action != 2:
            pyautogui.press(action_map[action])
        done, done_cap = self.get_terminated()
        new_observation = self.get_observation()
        reward = 1
        info = {}

        
        return new_observation, reward, done, False, info
        

    def reset(self,seed=None):
        self.seed = seed
        time.sleep(1)
        pyautogui.click(y=220, x=3280) 
        pyautogui.press('space')
        info = {}
        return self.get_observation(), info
        

    def render(self):
        pass

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:,:,:3]

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        channel = np.reshape(gray, (1,100,280))

        return channel
        

    def get_terminated(self):
        raw = np.array(self.cap.grab(self.done_location))[:,:,:3]

        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

        channel = np.reshape(gray, (1,20,220))
        channel = channel[0]

        done_strings = ['GAME', 'GAHE']

        done = False
        res = pytesseract.image_to_string(channel)[:4]
        if res in done_strings:
            done = True
            
        return done, channel
    def close(self):
        pass
    

env = WebGame()

model = DQN('CnnPolicy',env,buffer_size=400000)
load_path = os.path.join('models','DINOAI')
model = model.load(load_path)
evaluate_policy(model,env)
        

    
