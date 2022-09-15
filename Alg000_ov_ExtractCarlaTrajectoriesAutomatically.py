#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import sys

import time

try:
    sys.path.append(glob.glob('PythonAPI')[0])
except IndexError:
    print("PythonAPI package wasn't found")

try:
    sys.path.append(glob.glob('**/agents')[0])
except IndexError:
    print("PythonAPI/agents package wasn't found")

try:
    sys.path.append(glob.glob('**/agents/navigation')[0])
except IndexError:
    print("PythonAPI/agents/navigation package wasn't found")

try:
    sys.path.append(glob.glob('**/agents/tools')[0])
except IndexError:
    print("PythonAPI/agents/tools package wasn't found")

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
    
from Utils_Carla import Carla_loader
filePathWhereCarlaFolderIsNoted = 'ConfigurationFiles\CarlaFolder.txt'
Carla_loader.PrepareCarlaSysGivenPathToFileWhereCarlaFolderIsNoted(filePathWhereCarlaFolderIsNoted)

# ==============================================================================
# -- more imports --------------------------------------------------------------
# ==============================================================================

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error


start_point_attractor       = 74
#destination_point_attractor = 50

begin_attempt = 0
total_num_attempts = 200
number_of_changes_of_destination = 2

global collision_flag 

collision_flag = False

traffic_lights_on = True

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter   # <----- VEHICLE
        self._gamma = args.gamma
        self.restart(args) # <-------------------------------------------------------------- it is here that the player initial position is set!!!!
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        
        

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)
            
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
            
            
        ## ATTRACTOR ##########################################################
        # Spawn the player.
        print("Spawning the player")
        if self.player is not None:
            
            self.player.destroy()
            spawn_points = self.map.get_spawn_points()
            
            start_point_attractor = random.radint(0,len(spawn_points)-2)
            spawn_point = spawn_points[start_point_attractor]
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            #self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            start_point_attractor = random.randint(0,len(spawn_points)-1)
            spawn_point = spawn_points[start_point_attractor]                                 # <------------------------- START
            #spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        
            

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)
        
                    
    def traffic_light_set(self, clock):
        tl =self.player.get_traffic_light()
        if tl != None and tl.get_state() != carla.TrafficLightState.Green:
            tl.set_state(carla.TrafficLightState.Green)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))
            
    
    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================

import pickle
def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

import scipy.io as sio
from pathlib import Path



def imu_callback(imu,attempt_num, train):
            
    if train == True:
        nameFile = 'imu_train_' + str(attempt_num)
    else:
        nameFile = 'imu_test_' + str(attempt_num)
    
    print("imu measure:\n"+str(imu)+'\n')
    #accelerometer = imu.accelerometer.X
   # gyroscope     = imu.gyroscope.
    compass       = imu.compass
    data = np.zeros(1)
    #data[0] = accelerometer
    #data[1] = gyroscope
    data[0] = compass
    my_file = Path('C:/Users/asus/Documents/CARLA/WindowsNoEditor/PythonAPI/examples/position/'+ nameFile + '.npy')
    if my_file.is_file():
        positions = np.load('C:/Users/asus/Documents/CARLA/WindowsNoEditor/PythonAPI/examples/position/'+ nameFile + '.npy', allow_pickle = True)
        #positions = load_dict('/home/giulia/CARLA_last_version/PythonAPI/examples/positions.pckl')
        positions = positions.tolist()
    else:
        positions  =[]
    #positions = positions.tolist()
    positions.append(data)
    np.save('C:/Users/asus/Documents/CARLA/WindowsNoEditor/PythonAPI/examples/position/' + nameFile, positions)
    
    # Also save as matlab file
    sio.savemat(nameFile + '.mat', {'imus': positions})

    
def definePositionSensor(world, agent, attempt_num, train):
    
    # imu
    tick_time = 1.47
        
    imu_bp = world.world.get_blueprint_library().find('sensor.other.imu')
    imu_location = carla.Location(0,0,0)
    imu_rotation = carla.Rotation(0,0,0)
    imu_transform = carla.Transform(imu_location,imu_rotation)
    imu_bp.set_attribute("sensor_tick",str(tick_time))
    # add noise
    '''
    imu_bp.set_attribute("noise_gyro_stddev_x",str(0.1))
    imu_bp.set_attribute("noise_gyro_stddev_y",str(0.1))
    imu_bp.set_attribute("noise_gyro_stddev_z",str(0.1))
    imu_bp.set_attribute("noise_accel_stddev_x",str(0.1))
    imu_bp.set_attribute("noise_accel_stddev_y",str(0.1))
    imu_bp.set_attribute("noise_accel_stddev_z",str(0.1))
    '''
    
    ego_imu = world.world.spawn_actor(imu_bp,imu_transform,attach_to=agent.vehicle, attachment_type=carla.AttachmentType.Rigid)
    
    ego_imu.listen(lambda imu: imu_callback(imu,attempt_num, train))

    return ego_imu, imu_bp



       

def game_loop(train,random_less_velocity_attractor, number_of_changes_of_destination, args):
    """ Main loop for agent"""
    
    global baseDataFolderName
    
    global attempt_num
    
    startTime = time.time()
    
    pygame.init()
    pygame.font.init()
    world = None
    
    global positions_attractor
    positions_attractor = []
    global timeStamps
    timeStamps = []
    
    global path_to_imgs_current_attempt
    global path_to_positions_data
    path_to_imgs_current_attempt = baseDataFolderName + '/camera/' +'/%.4d' % attempt_num
    path_to_positions_data = baseDataFolderName + '/position/' 
    
    # Create folder if there is none
    if not os.path.exists(path_to_imgs_current_attempt):
        os.makedirs(path_to_imgs_current_attempt)
    if not os.path.exists(path_to_positions_data):
        os.makedirs(path_to_positions_data)
    # If it is not empty, empty it
    for f in os.listdir(path_to_imgs_current_attempt):
        os.remove(os.path.join(path_to_imgs_current_attempt, f))
    
    dist_attractor_dest_across_time = []
    
    destinations = []
    
    global previousTimeEvent
    previousTimeEvent = time.time()
    
    count = 0
    
    reached = False
    
    
    global collision_flag 
    collision_flag = False
    
    print('attempt num')
    print(attempt_num)
    
    current_destination_changes = 0
    
    traffic_lights_states = []
    
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(500000000.0)
        
        # This is to select the desired town
        client.load_world('Town02')
        client.reload_world()

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args)
        controller = KeyboardControl(world)
        
        # FIXED TIME STEP
        #settings = world.world.get_settings()
        #settings.fixed_delta_seconds = 0.05
        #world.world.apply_settings(settings)
        
       ################################àààà
        # sensors

        if args.agent == "Roaming":
            agent = RoamingAgent(world.player)
        elif args.agent == "Basic":
            agent = BasicAgent(world.player)
            spawn_point = world.map.get_spawn_points()[0]
            agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))
        else:
            
            ## ATTRACTOR
            agent = BehaviorAgent(world.player, behavior=args.behavior)
            spawn_points = world.map.get_spawn_points()

            # Printing all the spawn points so to know them and set them in a 
            # meaningful way
            spawn_points_list = []
            for i in range(len(spawn_points)):
                #print(i)
                #print(spawn_points[i])
                
                point = spawn_points[i]
                location_point   = point.location
                location_point_x = location_point.x
                location_point_y = location_point.y
                location_point_z = location_point.z
                location_point_array = np.zeros((1, 3))
                location_point_array[0, 0] = location_point_x
                location_point_array[0, 1] = location_point_y
                location_point_array[0, 2] = location_point_z
                
                spawn_points_list.append(location_point_array)
            
            
            destination_point_attractor = random.randint(0, len(spawn_points_list)-1)
            
            spawn_points_list = np.array(spawn_points_list)
            sio.savemat(path_to_positions_data + '/spawn_points' + '.mat', {'points': spawn_points_list})
            
            # Printing the chosen destination so to know which point we chose
            destination = spawn_points[destination_point_attractor].location # <---------------------------END
            destination_x = destination.x
            destination_y = destination.y
            destination_z = destination.z
            destination_array = np.zeros((1, 3))
            destination_array[0, 0] = destination_x
            destination_array[0, 1] = destination_y
            destination_array[0, 2] = destination_z
            sio.savemat(path_to_positions_data +'/destination' + '.mat', {'dest': destination_array})
            
            destinations.append(destination_array)
            sio.savemat(path_to_positions_data +'/destinations' + '.mat', {'dest': destinations})

            agent.set_destination(agent.vehicle.get_location(), destination, clean=True)
            
            #ego_imu, imu_bp = definePositionSensor(world, agent, attempt_num, train)
            
            cam_bp = None
            cam_bp = world.world.get_blueprint_library().find('sensor.camera.rgb')
            cam_bp.set_attribute("image_size_x",str(288))
            cam_bp.set_attribute("image_size_y",str(162))
            cam_bp.set_attribute("fov",str(90))
            cam_bp.set_attribute("sensor_tick",'1.0')
            cam_location = carla.Location(1,0, 1.5) # x, y, z; put z a bit higher than 0, or it will end up in the car's engine...
            cam_rotation = carla.Rotation(0,0,0) # (0, 0, 0) = camera facing forwards; (0, 180, 0) = camera facing backwardds
            cam_transform = carla.Transform(cam_location,cam_rotation)
            ego_cam = world.world.spawn_actor(cam_bp,cam_transform,
                                                       attach_to=agent.vehicle, attachment_type=carla.AttachmentType.Rigid)
               
            def function_handler_image_take(image, agent):
                
                global attempt_num
                global positions_attractor
                global timeStamps
                
                global currentTimeEvent
                global previousTimeEvent
                
                global path_to_imgs_current_attempt
                global path_to_positions_data
                
                currentTimeEvent = time.time()
                
                
                if currentTimeEvent - previousTimeEvent > 0.25:
                    
                    timeStamps.append(currentTimeEvent)
                
                    # Save the positions
                    nameFilePositions  = path_to_positions_data + '/real_position_train_' + '%.4d' % attempt_num
                    nameFileTimeStamps = path_to_positions_data + '/timestamps_train_' + '%.4d' % attempt_num
    
                    position = agent.vehicle.get_location()
                    position_x  = position.x
                    position_y  = position.y
                    position_z  = position.z
                    data = np.zeros((1,3))
                    data[0,0] = position_x # be careful to put also first 0 (this is not matlab)
                    data[0,1] = - position_y
                    data[0,2] = position_z
                    positions_attractor.append(data)
                    sio.savemat(nameFilePositions + '.mat', {'positions': positions_attractor})
                    sio.savemat(nameFileTimeStamps + '.mat', {'timeStamps': timeStamps})
                    
                    # Save the image
                    image.save_to_disk(path_to_imgs_current_attempt + '/%.4d.jpg' % image.frame)
                    
                    previousTimeEvent = time.time()
                
                return
            
            ego_cam.listen(lambda image: function_handler_image_take(image, agent))
            #ego_cam.listen(lambda image: image.save_to_disk(path_to_imgs_current_attempt + '/%.4d.jpg' % image.frame))
            
            def function_handler_collision(event):
                global collision_flag 
                collision_flag = True               
                return
            
            collision_sensor = world.world.spawn_actor(world.world.get_blueprint_library().find('sensor.other.collision'),
                                        carla.Transform(), attach_to=agent.vehicle)           
            collision_sensor.listen(lambda event: function_handler_collision(event))
            
        clock = pygame.time.Clock()
        
        while True:

            tick_time = 0.2
            clock.tick_busy_loop(20)

            if controller.parse_events():
                return

            # As soon as the server is ready continue!
            if not world.world.wait_for_tick(tick_time):
                continue

            if args.agent == "Roaming" or args.agent == "Basic":
                if controller.parse_events():
                    return

                # as soon as the server is ready continue!
                world.world.wait_for_tick(tick_time)

                world.tick(clock)
                world.render(display)
                pygame.display.flip()
                control = agent.run_step()
                control.manual_gear_shift = False
                world.player.apply_control(control)
                
            else:
                
                count += 1
                
                waypoint = world.map.get_waypoint(agent.vehicle.get_location()) # map can be taken from world object
                target   = world.map.get_waypoint(destination)

                dist_attractor_dest = waypoint.transform.location.distance(destination)
                dist_attractor_dest_across_time.append(dist_attractor_dest)
                
                traffic_lights_states.append(world.player.get_traffic_light())
                
                last_list_element = 100
                last_list_element_long_time = 5000
                
                #print('dist_attractor_destAttractor : '   +str(dist_attractor_dest))

                
                if (dist_attractor_dest < 25) :
                    
                    
                                      
                    if current_destination_changes < number_of_changes_of_destination:
                        
                        print('Intermediate destination reached')
                        
                        # Change the destination
                        # Printing the chosen destination so to know which point we chose
                        try:
                            print('Choosing new destination')
                            destination_point_attractor = random.randint(0, len(spawn_points_list)-1)
                            destination = spawn_points[destination_point_attractor].location
                            print('New destination chosen')
                            sio.savemat(path_to_positions_data + '/destination' + '.mat', {'dest': destination_array})
                            
                            destinations.append(destination_array)
                            sio.savemat(path_to_positions_data + '/destinations' + '.mat', {'dest': destinations})
                            agent.set_destination(agent.vehicle.get_location(), destination, clean=True)
                            print('New destination set')
                        except:
                            print('New destination not connected: looking for another one...')

                        current_destination_changes += 1
                        
                        endTime = time.time()
                        timeElapsedFromStartOfTrajectory = endTime - startTime
                        
                        print('Time passed from beginning of trajectory: ' + str(timeElapsedFromStartOfTrajectory))
                        
                    else:
                        
                        print('DESTINATION REACHED !!!!!')
                        print('RESET')
                        
                        print('Time passed from beginning of trajectory: ' + str(timeElapsedFromStartOfTrajectory))
                        
                        ego_cam.destroy()
                        world.destroy()    
                        #agent.destroy()
                        #ego_imu.destroy() 
                        
                        return
                
                if collision_flag == True:
                    
                    print('Carla detected a collision...')
                    print('RESET')
                    
                    attempt_num -= 1
                    
                    endTime = time.time()
                    timeElapsedFromStartOfTrajectory = endTime - startTime
                    
                    print('Time passed from beginning of trajectory: ' + str(timeElapsedFromStartOfTrajectory))
                    
                    ego_cam.destroy()
                    world.destroy()
                    #agent.destroy()
                    #ego_imu.destroy()
                    return
                
                if len(dist_attractor_dest_across_time) > last_list_element_long_time:
                    
                    if(abs(dist_attractor_dest - dist_attractor_dest_across_time[-last_list_element_long_time]) < 0.5 and
                       abs(dist_attractor_dest - dist_attractor_dest_across_time[-round(last_list_element_long_time/2)]) < 0.5):
                        
                        print('Seems like we had problem in the Carla path finder and we stopped...')
                        print('RESET')
                        
                        attempt_num -= 1
                        
                        endTime = time.time()
                        timeElapsedFromStartOfTrajectory = endTime - startTime
                        
                        print('Time passed from beginning of trajectory: ' + str(timeElapsedFromStartOfTrajectory))
                        
                        ego_cam.destroy()
                        world.destroy()
                        #agent.destroy()
                        #ego_imu.destroy() 
                        return
                        
                
                if len(dist_attractor_dest_across_time) > last_list_element:
                    if(abs(dist_attractor_dest - dist_attractor_dest_across_time[-last_list_element]) < 0.5 and
                       abs(dist_attractor_dest - dist_attractor_dest_across_time[-round(last_list_element/2)]) < 0.5):
                       #and traffic_lights_states[-last_list_element] != None
                       #and traffic_lights_states[-round(last_list_element/2)] != None
                       #and traffic_lights_states[-1] != None
                       #and traffic_lights_states[-last_list_element].get_state() == carla.TrafficLightState.Green 
                       #and traffic_lights_states[-round(last_list_element/2)].get_state() == carla.TrafficLightState.Green
                       #and traffic_lights_states[-1].get_state() == carla.TrafficLightState.Green):
                    
                        print('Seems like we crashed or got stuck somewhere...')
                        print('RESET')
                        
                        attempt_num -= 1
                        
                        endTime = time.time()
                        timeElapsedFromStartOfTrajectory = endTime - startTime
                        
                        print('Time passed from beginning of trajectory: ' + str(timeElapsedFromStartOfTrajectory))
                        
                        ego_cam.destroy()
                        world.destroy()
                        return
                

                    
                world.tick(clock)
                if traffic_lights_on == True:
                    world.traffic_light_set(clock)
                world.render(display)
                pygame.display.flip()
                
                speed_limit          = world.player.get_speed_limit()
                
                if attempt_num != 0:
                    
                    velocity_attractor = max(15, (speed_limit-random_less_velocity_attractor))
                    #velocity_attractor = min(velocity_attractor, 70)
                    
                    #if velocity_attractor > 50:
                    #    velocity_attractor = velocity_attractor/2
                    
                    agent.update_information(velocity_attractor, world.player)
                    
                    agent.get_local_planner().set_speed(velocity_attractor)
                    
                    
                else:
                    
                    velocity_attractor = max(15, (speed_limit-random_less_velocity_attractor))
                    #velocity_attractor = min(velocity_attractor, 70)
                    
                    #if velocity_attractor > 50:
                    #    velocity_attractor = velocity_attractor/2
                    
                    agent.update_information(velocity_attractor, world.player)
                    
                    agent.get_local_planner().set_speed(velocity_attractor)
                    
                    '''
                    speed_limit = min(speed_limit, 10)
                    agent.update_information(speed_limit, world.player)
                    agent.get_local_planner().set_speed(speed_limit)
                    '''
                
                control_attractor = agent.run_step()
                world.player.apply_control(control_attractor)

    finally:
        #if world is not None:
        #    world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.audi.a2', # select car (choose a small one so it does not appear)
        help='Actor filter (default: "vehicle.audi.a2")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='l',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        #choices=["cautious", "normal", "aggressive"],
        choices=["cautious", "normal"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='cautious')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    
    # -.--.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.--.-.-.-.-  
    # Before starting the game loop, let us extract the name of the folder
    # where to save everything
    global baseDataFolderName
    with open('ConfigurationFiles\BaseDataFolder.txt') as f:
        baseDataFolderName = f.readlines()[0]
        
    # -.--.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.--.-.-.-.-  

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try: 
        
        global attempt_num       

        train = True

        attempt_num = begin_attempt
        
        while(attempt_num < total_num_attempts):
            
            random_less_velocity_attractor = random.randint(0, 15)
            
            game_loop(train,random_less_velocity_attractor, number_of_changes_of_destination, args)
            attempt_num += 1


    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
