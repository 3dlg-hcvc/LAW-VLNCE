import gzip
import json
from typing import Any, List

import torch
import torch.nn.functional as F
import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from habitat.config import Config
from habitat.core.embodied_task import (
    EmbodiedTask,
    Measure,
    SimulatorTaskAction,
)
from habitat.core.dataset import Dataset, Episode
from habitat.core.registry import registry
from habitat.core.simulator import (
    AgentState,
    RGBSensor,
    Sensor,
    SensorTypes,
    ShortestPathPoint,
    Simulator,
)
from habitat.utils.visualizations import fog_of_war, maps
from habitat.core.utils import not_none_validator, try_cv2_import
from habitat.tasks.utils import cartesian_to_polar
from habitat.tasks import utils
from vlnce_baselines.common.utils import quaternion_rotate_vector

cv2 = try_cv2_import()

# MAP_THICKNESS_SCALAR= 1250
MAP_THICKNESS_SCALAR= 125


@registry.register_measure
class PathLength(Measure):
    r"""Path Length (PL)

    PL = sum(geodesic_distance(agent_prev_position, agent_position)
            over all agent positions.
    """

    def __init__(self, sim: Simulator, config: Config, *args: Any, **kwargs: Any):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = self._sim.geodesic_distance(
            self._previous_position, episode.goals[0].position
        )
        self._agent_episode_distance = 0.0
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position

        self._metric = self._agent_episode_distance

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "path_length"


@registry.register_measure
class OracleNavigationError(Measure):
    r"""Oracle Navigation Error (ONE)

    ONE = min(geosdesic_distance(agent_pos, goal))
            over all agent_pos in agent path.

    This computes oracle navigation error for every update regardless of
    whether or not the end of the episode has been reached.
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = float("inf")

    def update_metric(self, *args: Any, episode, action, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if distance_to_target < self._metric:
            self._metric = distance_to_target

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_navigation_error"

@registry.register_measure
class OracleSuccess(Measure):
    r"""Oracle Success Rate (OSR)

    OSR = I(ONE <= goal_radius),
    where ONE is Oracle Navigation Error.
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        if self._metric:
            # skip, already had oracle success
            return

        current_position = self._sim.get_agent_state().position.tolist()
        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if distance_to_target < self._config.SUCCESS_DISTANCE:
            self._metric = 1

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_success"


@registry.register_measure
class OracleSPL(Measure):
    r"""OracleSPL (Oracle Success weighted by Path Length)

    OracleSPL = max(SPL) over all points in the agent path
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._ep_success = None
        self._sim = sim
        self._config = config
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_episode_distance = episode.info["geodesic_distance"]
        self._agent_episode_distance = 0.0
        self._ep_success = 0
        self._metric = 0.0

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        if self._ep_success:  # shortest path already found
            return

        current_position = self._sim.get_agent_state().position.tolist()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if distance_to_target < self._config.SUCCESS_DISTANCE:
            self._ep_success = 1
            self._metric = self._ep_success * (
                self._start_end_episode_distance
                / max(self._start_end_episode_distance, self._agent_episode_distance)
            )

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "oracle_spl"


@registry.register_measure
class StepsTaken(Measure):
    r"""Counts the number of times update_metric() is called. This is equal to
    the number of times that the agent takes an action. STOP counts as an
    action.
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._metric = 0
        super().__init__()

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = 0

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        self._metric += 1

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "steps_taken"


@registry.register_measure
class NDTW(Measure):
    r"""NDTW (Normalized Dynamic Time Warping)

    ref: Effective and General Evaluation for Instruction
        Conditioned Navigation using Dynamic Time
        Warping - Magalhaes et. al
    https://arxiv.org/pdf/1907.05446.pdf
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.locations = []
        self.gt_locations = []
        self.dtw_func = fastdtw if config.FDTW else dtw

        gt_path = config.GT_PATH.format(split=config.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_json = json.load(f)
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "ndtw"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations.clear()
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position == self.locations[-1]:
                return
            self.locations.append(current_position)

        dtw_distance = self.dtw_func(
            self.locations, self.gt_locations, dist=self._euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE)
        )
        self._metric = nDTW


@registry.register_measure
class SDTW(Measure):
    r"""SDTW (Success Weighted be nDTW)

    ref: Effective and General Evaluation for Instruction
        Conditioned Navigation using Dynamic Time
        Warping - Magalhaes et. al
    https://arxiv.org/pdf/1907.05446.pdf
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.locations = []
        self.gt_locations = []
        self.dtw_func = fastdtw if config.FDTW else dtw

        gt_path = config.GT_PATH.format(split=config.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_json = json.load(f)
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "sdtw"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.locations.clear()
        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self._metric = None

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        if len(self.locations) == 0:
            self.locations.append(current_position)
        else:
            if current_position != self.locations[-1]:
                self.locations.append(current_position)

        dtw_distance = self.dtw_func(
            self.locations, self.gt_locations, dist=self._euclidean_distance
        )[0]

        nDTW = np.exp(
            -dtw_distance / (len(self.gt_locations) * self._config.SUCCESS_DISTANCE)
        )

        distance_to_target = self._sim.geodesic_distance(
            current_position, episode.goals[0].position
        )
        if task.is_stop_called and distance_to_target < self._config.SUCCESS_DISTANCE:
            ep_success = 1
        else:
            ep_success = 0

        self._metric = ep_success * nDTW

@registry.register_measure
class WaypointAccuracy(Measure):
    r"""Waypoint Accuracy
    """

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.success_distance = getattr(config, "SUCCESS_DISTANCE", 0.5)
        self.gt_locations = []
        self.correct_ways = []
        super().__init__()

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "way_accuracy"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self.gt_locations = episode.reference_path
        self.correct_ways = [0 for _ in self.gt_locations]
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        current_position = self._sim.get_agent_state().position.tolist()
        
        for ind, way in enumerate(self.gt_locations):
            distance_to_way = self._sim.geodesic_distance(
                current_position, way
            )
            if distance_to_way < self.success_distance:
                self.correct_ways[ind] = 1

        num_correct_way = 0
        for p in self.correct_ways:
            num_correct_way += p

        self._metric = (num_correct_way / len(self.gt_locations))

@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count = None
        self._map_resolution = (config.MAP_RESOLUTION, config.MAP_RESOLUTION)
        self._num_samples = config.NUM_TOPDOWN_MAP_SAMPLE_POINTS
        self._ind_x_min = None
        self._ind_x_max = None
        self._ind_y_min = None
        self._ind_y_max = None
        self._previous_xy_location = None
        self._previous_agent_location = None
        self._previous_agent_loc_point_types = None
        self._coordinate_min = maps.COORDINATE_MIN
        self._coordinate_max = maps.COORDINATE_MAX
        self._top_down_map = None
        self._shortest_path_points = None
        self._cell_scale = (
            self._coordinate_max - self._coordinate_min
        ) / self._map_resolution[0]
        self.line_thickness = int(
            np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
        )

        self.gt_locations = []
        gt_path = config.GT_PATH.format(split=config.SPLIT)
        with gzip.open(gt_path, "rt") as f:
            self.gt_json = json.load(f)
        self.prev_top_down_map = None
        self.prev_point_oracle_way = 0
        self.prev_point_pred_way = 0
        self.prev_point_oracle_reconv = 0
        self.agent_coords_list = []
        self.agent_rotation_list = []

        self.MAP_AGENT_BOUNDING_BOX = 10
        self.MAP_ORACLE_WAY = 11
        self.MAP_PRED_WAY_CORRECT = 12
        self.MAP_PRED_WAY_SOMEWHAT_CORRECT = 97
        self.MAP_PRED_WAY_INCORRECT = 13
        self.MAP_PRED_PATH_COLOR = 14
        self.MAP_ORACLE_RECONV = 15
        self.crop_map_size = 2.2

        super().__init__()

    def _check_valid_nav_point(self, point: List[float]):
        self._sim.is_navigable(point)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(np.array(position_b) - np.array(position_a), ord=2)

    def get_original_map(self):
        top_down_map = maps.get_topdown_map(
            self._sim,
            self._map_resolution,
            self._num_samples,
            self._config.DRAW_BORDER,
        )

        range_x = np.where(np.any(top_down_map, axis=1))[0]
        range_y = np.where(np.any(top_down_map, axis=0))[0]

        self._ind_x_min = range_x[0]
        self._ind_x_max = range_x[-1]
        self._ind_y_min = range_y[0]
        self._ind_y_max = range_y[-1]

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)

        return top_down_map

    def draw_source_and_target(self, episode):
        # mark source point
        s_x, s_y = maps.to_grid(
            episode.start_position[0],
            episode.start_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        # point_padding = 2 * int(
        #     np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        # )

        point_padding = 0

        # self._top_down_map[
        #     s_x - point_padding : s_x + point_padding + 1,
        #     s_y - point_padding : s_y + point_padding + 1,
        # ] = maps.MAP_SOURCE_POINT_INDICATOR

        # mark target point
        t_x, t_y = maps.to_grid(
            episode.goals[0].position[0],
            episode.goals[0].position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._top_down_map[
            t_x - point_padding : t_x + point_padding + 1,
            t_y - point_padding : t_y + point_padding + 1,
        ] = maps.MAP_TARGET_POINT_INDICATOR
    
    def _draw_point(self, position, point_type, _map):
        
        point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        point_padding = 0
        t_x, t_y = maps.to_grid(
            position[0],
            position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        
        _map[
            t_x - point_padding : t_x + point_padding + 1,
            t_y - point_padding : t_y + point_padding + 1,
        ] = point_type
        
        return _map
    
    def _draw_point_with_text(self, position, point_type, _map, display_str):
        
        point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        t_x, t_y = maps.to_grid(
            position[0],
            position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        
        _map[
            t_x - point_padding : t_x + point_padding + 1,
            t_y - point_padding : t_y + point_padding + 1,
        ] = point_type

        cv2.putText(_map, display_str, (t_x, t_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 3)
        
        return _map

    def _undraw_point(self, position, point_type):
        point_padding = 2 * int(
            np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
        )
        point_padding = 0
        t_x, t_y = maps.to_grid(
            position[0],
            position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._top_down_map[
            t_x - point_padding : t_x + point_padding + 1,
            t_y - point_padding : t_y + point_padding + 1,
        ] = point_type


    def pol2cart(self, theta, rho, x1, y1):
        x = x1 + rho * np.cos(theta)
        y = y1 - rho * np.sin(theta)
        return x, y

    def _draw_agent_pred_border(self, agent_position, agent_rot, _map):
        try:
            heading = self.get_polar_angle()

            point_padding = 2 * int(
                np.ceil(self._map_resolution[0] / MAP_THICKNESS_SCALAR)
            )
            point_padding = 0
            corners = [
                agent_position + np.array([x, 0, z])
                for x, z in [
                    (-self.crop_map_size, -self.crop_map_size),
                    (-self.crop_map_size, self.crop_map_size),
                    (self.crop_map_size, self.crop_map_size),
                    (self.crop_map_size, -self.crop_map_size),
                ]
            ]
            
####
            grid_corners = [
                maps.to_grid(
                    c[0],
                    c[2],
                    self._coordinate_min,
                    self._coordinate_max,
                    self._map_resolution,
                ) for c in corners
            ]

            agent_center = maps.to_grid(
                    agent_position[0],
                    agent_position[2],
                    self._coordinate_min,
                    self._coordinate_max,
                    self._map_resolution,
                )


            theta = agent_rot - np.pi/2
            rho = np.linalg.norm(
                np.array(agent_center) - np.array(grid_corners[0]), ord=2
            )

            pt1 = agent_center[::-1]
            c1 = self.pol2cart(theta -np.pi/4, rho, pt1[0], pt1[1])
            c2 = self.pol2cart(theta -(3/4)*np.pi, rho, pt1[0], pt1[1])
            c3 = self.pol2cart(theta +(3/4)*np.pi, rho, pt1[0], pt1[1])
            c4 = self.pol2cart(theta +np.pi/4, rho, pt1[0], pt1[1])

            draw_corners = [c1, c2, c3, c4]

            pts = np.array([c1, c2, c3, c4], np.int32)
            pts = pts.reshape((-1,1,2))

            cv2.drawContours(_map, [pts], -1, self.MAP_AGENT_BOUNDING_BOX, 1)

        except AttributeError:
            pass
    
        return _map


    def _draw_waypoints_predicted(self, waypoint, oracle_way_location, waypoint_location_grid, oracle_way_location_grid, oracle_reconverted, _map):
        if self._config.DRAW_WAYPOINTS:
            try:
                if waypoint_location_grid is not None and oracle_way_location_grid is not None:
                    distance_way_to_gt = self._euclidean_distance(
                        waypoint_location_grid, oracle_way_location_grid
                    )
                    if distance_way_to_gt <= self._config.SUCCESS_DISTANCE:
                        self.prev_point_pred_way = self._draw_point(
                            waypoint,
                            self.MAP_PRED_WAY_CORRECT,
                            _map
                        )
                    else:
                            self.prev_point_pred_way = self._draw_point(
                            waypoint,
                            self.MAP_PRED_WAY_INCORRECT,
                            _map
                        )
                
                if oracle_way_location is not None:
                    self.prev_point_oracle_way = self._draw_point(
                            oracle_way_location,
                            self.MAP_ORACLE_WAY,
                            _map
                        )

                if oracle_reconverted is not None:
                    self.prev_point_oracle_reconv = self._draw_point(
                            oracle_reconverted,
                            self.MAP_ORACLE_RECONV,
                            _map
                        )
                
            except AttributeError:
                pass

        return _map

    def _draw_waypoints_oracle(self, oracle_way_location, episode, _map):
        if self._config.DRAW_WAYPOINTS:
            try:
                for way in episode.reference_path:
                    self._draw_point_with_text(
                        way,
                        self.MAP_ORACLE_WAY,
                        _map,
                        "1"
                    )

            except AttributeError:
                pass

        return _map

    def _draw_gt_shortest_path_through_waypoints(self, episode):

        agent_position = self._sim.get_agent_state().position

        shortest_path_points =  []
        for ind, gt in enumerate(self.gt_locations):
            if (ind % 5) == 0:
                next_points = self._sim.get_straight_shortest_path_points(
                    agent_position, gt)
                for p in next_points:
                    shortest_path_points.append(p)

        shortest_path_points = np.unique(shortest_path_points, axis=0)
        return shortest_path_points

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._step_count = 0
        self._metric = None

        self.gt_locations = self.gt_json[str(episode.episode_id)]["locations"]
        self.prev_top_down_map = None
        self.prev_point_oracle_way = 0
        self.prev_point_pred_way = 0
        self.prev_point_oracle_reconv = 0
        self.agent_coords_list = []
        self.agent_rotation_list = []

        self._top_down_map = self.get_original_map()

        # store previous agent pos
        agent_position = self._sim.get_agent_state().position
        self._previous_agent_location = agent_position
        corners = [
            agent_position + np.array([x, 0, z])
            for x, z in [
                (-self.crop_map_size, -self.crop_map_size),
                (-self.crop_map_size, self.crop_map_size),
                (self.crop_map_size, self.crop_map_size),
                (self.crop_map_size, -self.crop_map_size),
            ]
        ]
        map_corners = [
                maps.to_grid(
                    p[0],
                    p[2],
                    self._coordinate_min,
                    self._coordinate_max,
                    self._map_resolution,
                )
                for p in corners
            ]
        self._previous_agent_loc_point_types = []
        for ind, c in enumerate(map_corners):
            self._previous_agent_loc_point_types.append(self._top_down_map[c])

        # END - store previous agent pos

        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        self._previous_xy_location = (a_y, a_x)
        if self._config.DRAW_SHORTEST_PATH:
            # draw shortest path
            self._shortest_path_points = self._draw_gt_shortest_path_through_waypoints(episode)
            self._shortest_path_points = [
                maps.to_grid(
                    p[0],
                    p[2],
                    self._coordinate_min,
                    self._coordinate_max,
                    self._map_resolution,
                )[::-1]
                for p in self._shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target points last to avoid overlap
        if self._config.DRAW_SOURCE_AND_TARGET:
            self.draw_source_and_target(episode)

    def _clip_map(self, _map):
        return _map[
            self._ind_x_min
            - self._grid_delta : self._ind_x_max
            + self._grid_delta,
            self._ind_y_min
            - self._grid_delta : self._ind_y_max
            + self._grid_delta,
        ]

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1

        house_map, map_agent_x, map_agent_y, agent_position = self.update_map(
                self._sim.get_agent_state().position
            )

        if action is not None:
            if "waypoint_locs" in action:
                waypoint_location = action["waypoint_locs"]
            else:
                waypoint_location = None

            if "oracle_way" in action:
                oracle_way_location = action["oracle_way"]
            else:
                oracle_way_location = None


            if "waypoint_locs_grid" in action:
                waypoint_location_grid = action["waypoint_locs_grid"]
            else:
                waypoint_location_grid = None
            if "oracle_way_grid" in action:
                oracle_way_location_grid = action["oracle_way_grid"]
            else:
                oracle_way_location_grid = None
            if "oracle_conv" in action:
                oracle_conv = action["oracle_conv"]
            else:
                oracle_conv = None

            #self._top_down_map = house_map.copy()

            # draw prediction and oracle
            #house_map = self._draw_waypoints_predicted(waypoint_location, oracle_way_location, waypoint_location_grid, oracle_way_location_grid, oracle_conv, house_map)
            house_map = self._draw_waypoints_oracle(oracle_way_location, episode, house_map)

            # draw prediction rectangle
            #agent_rot = self.get_polar_angle()
            #house_map = self._draw_agent_pred_border(agent_position, agent_rot, house_map)

        # Rather than return the whole map which may have large empty regions,
        # only return the occupied part (plus some padding).
        clipped_house_map = self._clip_map(house_map)

        clipped_fog_of_war_map = None
        if self._config.FOG_OF_WAR.DRAW:
            clipped_fog_of_war_map = self._clip_map(self._fog_of_war_mask)

        self.agent_coords_list = []
        self.agent_coords_list.append((
            map_agent_x - (self._ind_x_min - self._grid_delta),
            map_agent_y - (self._ind_y_min - self._grid_delta),
        ))

        self._metric = {
            "map": clipped_house_map,
            "fog_of_war_mask": clipped_fog_of_war_map,
            "agent_map_coord": (
                map_agent_x - (self._ind_x_min - self._grid_delta),
                map_agent_y - (self._ind_y_min - self._grid_delta),
            ),
            "agent_angle": self.get_polar_angle(),
        }

    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

    def rotate_coords(self, coords, heading):
        coords = torch.tensor([coords], dtype=torch.float)
        heading = torch.tensor([heading])
        sin_t = torch.sin(heading)
        cos_t = torch.cos(heading)
        A = torch.zeros(2, 2, dtype=torch.float)
        A[0, 0] = cos_t
        A[0, 1] = sin_t
        A[1, 0] = -sin_t
        A[1, 1] = cos_t

        rotated_coords = torch.mm(coords, A).squeeze(0).cpu().numpy()

        return rotated_coords

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[0],
            agent_position[2],
            self._coordinate_min,
            self._coordinate_max,
            self._map_resolution,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = int(
                np.round(self._map_resolution[0] * 2 / MAP_THICKNESS_SCALAR)
            )
            thickness = 1
            
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                29,
                thickness=thickness,
            )
        
        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw prediction rectangle
        #self._draw_agent_pred_border(agent_position)

        self._previous_xy_location = (a_y, a_x)
        self._previous_agent_location = agent_position
        return self._top_down_map, a_x, a_y, agent_position

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                * max(self._map_resolution)
                / (self._coordinate_max - self._coordinate_min),
            )

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return "top_down_map"