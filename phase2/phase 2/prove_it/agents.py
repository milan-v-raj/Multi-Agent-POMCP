# agents.py
import pygame
import numpy as np
import random
from config import *

class Drone(pygame.sprite.Sprite):
    def __init__(self, x, y, color, speed_limit=5.0):
        super().__init__()
        self.radius = 10
        self.image = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, color, (self.radius, self.radius), self.radius)
        self.rect = self.image.get_rect(center=(x, y))
        
        self.color = color
        self.max_speed = speed_limit
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        self.acc = np.array([0.0, 0.0])

    def apply_force(self, force):
        self.acc += force

    def update(self):
        # Kinematics
        self.vel += self.acc
        self.vel *= FRICTION
        
        # Speed Limiting
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = (self.vel / speed) * self.max_speed
            
        self.pos += self.vel
        self.rect.center = self.pos
        self.acc = np.array([0.0, 0.0]) # Reset acceleration

        # Boundary checks (Outer walls)
        if self.pos[0] < self.radius: self.pos[0] = self.radius; self.vel[0] *= -0.5
        if self.pos[0] > WIDTH - self.radius: self.pos[0] = WIDTH - self.radius; self.vel[0] *= -0.5
        if self.pos[1] < self.radius: self.pos[1] = self.radius; self.vel[1] *= -0.5
        if self.pos[1] > HEIGHT - self.radius: self.pos[1] = HEIGHT - self.radius; self.vel[1] *= -0.5

    def check_collision(self, arena):
        # Slide against interior walls
        for wall in arena.walls:
            if self.rect.colliderect(wall.rect):
                clip = self.rect.clip(wall.rect)
                if clip.width < clip.height:
                    self.vel[0] *= -0.5 # Bounce slightly
                    if self.rect.centerx < wall.rect.centerx: self.pos[0] -= clip.width
                    else: self.pos[0] += clip.width
                else:
                    self.vel[1] *= -0.5 # Bounce slightly
                    if self.rect.centery < wall.rect.centery: self.pos[1] -= clip.height
                    else: self.pos[1] += clip.height
                self.rect.center = self.pos

class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y, vx, vy):
        super().__init__()
        self.radius = 4
        self.image = pygame.Surface((self.radius*2, self.radius*2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (255, 50, 50, 60), (self.radius, self.radius), self.radius) # Faint red
        self.rect = self.image.get_rect(center=(x, y))
        
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([float(vx), float(vy)])

    def update(self):
        self.pos += self.vel
        # Add Brownian motion (uncertainty growth over time)
        self.pos += np.random.normal(0, 1.0, 2) 
        self.rect.center = self.pos

class ParticleFilter:
    def __init__(self, num_particles=150):
        self.num = num_particles
        self.particles = pygame.sprite.Group()
        self.is_initialized = False

    def predict(self):
        # Move all particles according to their velocity and noise
        for p in self.particles:
            p.update()

    def update(self, observation, target_vel):
        # If we see the target, snap the belief cloud back to reality
        if observation is not None:
            self.particles.empty()
            self.is_initialized = True
            obs_x, obs_y = observation
            vel_x, vel_y = target_vel 
            
            # Resample particles around the known observation
            for _ in range(self.num): 
                px = obs_x + np.random.normal(0, 5)
                py = obs_y + np.random.normal(0, 5)
                self.particles.add(Particle(px, py, vel_x, vel_y))

    def get_belief_center_and_spread(self):
        # Returns the (Center_Coordinate, Spread_Radius)
        if len(self.particles) == 0:
            return np.array([WIDTH/2, HEIGHT/2]), 0.0
            
        all_pos = np.array([p.pos for p in self.particles])
        center = np.mean(all_pos, axis=0)
        spread = np.std(all_pos)
        return center, spread