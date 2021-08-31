from typing import Any
import gzip
import json
import pickle
import h5py
import os
import numpy as np
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.utils import (
    cartesian_to_polar,
)
from habitat_extensions.shortest_path_follower import ShortestPathFollower
import geometer as geometer

def quaternion_rotate_vector(quat: np.quaternion, v: np.array) -> np.array:
    r"""Rotates a vector by a quaternion

    Args:
        quaternion: The quaternion to rotate by
        v: The vector to rotate

    Returns:
        np.array: The rotated vector
    """
    vq = np.quaternion(0, 0, 0, 0)
    vq.imag = v
    return (quat * vq * quat.inverse()).imag

@registry.register_sensor(name="GlobalGPSSensor")
class GlobalGPSSensor(Sensor):
    r"""The agents current location in the global coordinate frame

    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions
                to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "globalgps"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(self._dimensionality,),
            dtype=np.float32,
        )

    def get_observation(self, *args: Any, observations, episode, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        return agent_position.astype(np.float32)


@registry.register_sensor
class VLNOracleActionSplineSensor(Sensor):
    r"""Sensor for observing the optimal action to take. Does not rely on the shortest path to the Goal.
    Instead observes the next waypoint (nearest waypoint on the GT trajectory spline) 
    and the best action towards it based on the shortest path distance.
    Maintains a visitation for the waypoints
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        gt_path = config.GT_PATH.format(split=config.SPLIT)

        self.is_sparse = getattr(config, "IS_SPARSE", True)
        self.num_inter_waypoints = getattr(config, "NUM_WAYPOINTS", 0)   # Set number of intermediate waypoints
        
        with gzip.open(gt_path, "rt") as f:
            self.gt_waypoint_locations = json.load(f)

        super().__init__(config=config)
        self.follower = ShortestPathFollower(
            self._sim,
            # all goals can be navigated to within 0.5m.
            goal_radius=getattr(config, "GOAL_RADIUS", 0.5),
            return_one_hot=False,
        )
        self.follower.mode = "geodesic_path"

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "vln_oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        way_locations = episode.reference_path
        way_loc_points = [geometer.Point(x, y, z) for x, y, z in way_locations]
        trajectory_lines = [geometer.Line(way_loc_points[ind], way_loc_points[ind+1]) for ind in range(len(way_loc_points)-1)]
        trajectory_segments = [geometer.shapes.Segment(way_loc_points[ind], way_loc_points[ind+1]) for ind in range(len(way_loc_points)-1)]
        
        current_position = self._sim.get_agent_state().position.tolist()
        current_position_point = geometer.Point(current_position[0], current_position[1], current_position[2])

        next_way = way_locations[0]
        nearest_way = 0
        for ind, line in enumerate(trajectory_lines):
            nearest_way = (ind + 1) if ind < (len(trajectory_lines)-1) else (len(trajectory_lines)-1)
            next_way = way_locations[nearest_way]      # the end point of the line segment
            if trajectory_segments[ind].contains(current_position_point):
                break

            current_position_projected = line.project(current_position_point)
            if trajectory_segments[ind].contains(current_position_projected):
                break

        best_action = self.follower.get_next_action(next_way)

        if best_action in [None, HabitatSimActions.STOP]:
            while best_action in [None, HabitatSimActions.STOP]:
                # if the agent has reached the current waypoint then update the next waypoint
                if nearest_way == (len(trajectory_lines)-1):
                    return np.array([HabitatSimActions.STOP])
                
                nearest_way = nearest_way + 1
                next_way = way_locations[nearest_way]
                best_action = self.follower.get_next_action(next_way)
            
            return np.array([best_action])

        return np.array([best_action])



@registry.register_sensor
class VLNOracleActionGeodesicSensor(Sensor):
    r"""Sensor for observing the optimal action to take. Does not rely on the shortest path to the Goal.
    Instead observes the next waypoint and the best action towards that based on the shortest path.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        gt_path = config.GT_PATH.format(split=config.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_waypoint_locations = json.load(f)
        
        self.is_sparse = getattr(config, "IS_SPARSE", True)
        self.num_inter_waypoints = getattr(config, "NUM_WAYPOINTS", 0)   # Set number of intermediate waypoints
        self.goal_radius = getattr(config, "GOAL_RADIUS", 0.5)

        super().__init__(config=config)
        self.follower = ShortestPathFollower(
            self._sim,
            # all goals can be navigated to within 0.5m.
            goal_radius=getattr(config, "GOAL_RADIUS", 0.5),
            return_one_hot=False,
        )
        self.follower.mode = "geodesic_path"
        self.reference_path_action = None
        self.next_way_action = 0
        self.episode_id_action = None

        self.possible_actions= [HabitatSimActions.MOVE_FORWARD, HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "vln_oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        
        if self.num_inter_waypoints > 0:
            locs = self.gt_waypoint_locations[str(episode.episode_id)]["locations"] #episode.reference_path
            ep_path_length = self._sim.geodesic_distance(locs[0], episode.goals[0].position)
            
            way_locations = [locs[0]]
            count = 0
            dist = ep_path_length / (self.num_inter_waypoints+1)
            for way in locs[:-1]:
                d = self._sim.geodesic_distance(locs[0], way)
                if d >= dist:
                    way_locations.append(way)
                    if count >= (self.num_inter_waypoints-1):
                        break
                    count += 1
                    dist += ep_path_length / (self.num_inter_waypoints+1)

            way_locations.append(episode.goals[0].position)

        else:
            if self.is_sparse:
                # Sparse supervision of waypoints
                way_locations = episode.reference_path
            else:
                # Dense supervision of waypoints
                way_locations = self.gt_waypoint_locations[str(episode.episode_id)]["locations"]

        current_position = self._sim.get_agent_state().position.tolist()
        
        nearest_dist = float("inf")
        nearest_way = way_locations[-1]
        nearest_way_count = len(way_locations)-1
        for ind, way in reversed(list(enumerate(way_locations))):
            distance_to_way = self._sim.geodesic_distance(
                current_position, way
            )
            if distance_to_way > self.goal_radius and distance_to_way < nearest_dist:
                dist_way_to_goal = self._sim.geodesic_distance(
                    way, episode.goals[0].position
                )
                dist_agent_to_goal = self._sim.geodesic_distance(
                    current_position, episode.goals[0].position
                )
                if dist_agent_to_goal > dist_way_to_goal:
                    nearest_dist = distance_to_way
                    nearest_way = way
                    nearest_way_count = ind

        best_action = self.follower.get_next_action(nearest_way)

        if best_action is None:
            while best_action is None:
                # if the agent has reached the current waypoint then update the next waypoint
                if nearest_way_count == (len(way_locations)-1):
                    return np.array([HabitatSimActions.STOP])
                
                nearest_way_count = nearest_way_count + 1
                nearest_way = way_locations[nearest_way_count]
                best_action = self.follower.get_next_action(nearest_way)
            
            return np.array([best_action])
            
        return np.array([best_action])

@registry.register_sensor
class VLNOracleActionSensor(Sensor):
    r"""Sensor used in the VLNCE paper. Sensor for observing the optimal action to take. The assumption this
    sensor currently makes is that the shortest path to the goal is the
    optimal path.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        super().__init__(config=config)

        # all goals can be navigated to within 0.5m.
        goal_radius = getattr(config, "GOAL_RADIUS", 0.5)
        self.follower = ShortestPathFollower(
                sim, goal_radius, return_one_hot=False
            )
        self.follower.mode = "geodesic_path"

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "vln_oracle_action_sensor"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=100, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        best_action = self.follower.get_next_action(episode.goals[0].position)
        return np.array(
            [best_action if best_action is not None else HabitatSimActions.STOP]
        )
        
@registry.register_sensor
class VLNOracleProgressSensor(Sensor):
    r"""Sensor for observing how much progress has been made towards the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "progress"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        # TODO: what is the correct sensor type?
        return SensorTypes.MEASUREMENT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        distance_from_start = episode.info["geodesic_distance"]

        return (distance_from_start - distance_to_target) / distance_from_start

