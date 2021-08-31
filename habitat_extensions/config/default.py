from typing import List, Optional, Union

from habitat.config.default import Config, get_config
from yacs.config import CfgNode as CN

_C = get_config()
_C.defrost()

# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GLOBAL_GPS_SENSOR = Config()
_C.TASK.GLOBAL_GPS_SENSOR.TYPE = "GlobalGPSSensor"
_C.TASK.GLOBAL_GPS_SENSOR.DIMENSIONALITY = 3
# -----------------------------------------------------------------------------
# VLN ORACLE ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_ACTION_SENSOR = Config()
_C.TASK.VLN_ORACLE_ACTION_SENSOR.TYPE = "VLNOracleActionSensor"
_C.TASK.VLN_ORACLE_ACTION_SENSOR.GOAL_RADIUS = 0.5
# compatibility with the dataset generation oracle and paper results.
# if False, use the ShortestPathFollower in Habitat
_C.TASK.VLN_ORACLE_ACTION_SENSOR.USE_ORIGINAL_FOLLOWER = True
_C.TASK.VLN_ORACLE_ACTION_SENSOR.SPLIT = "train"
_C.TASK.VLN_ORACLE_ACTION_SENSOR.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz"
)
_C.TASK.VLN_ORACLE_ACTION_SENSOR.IS_SPARSE = True
_C.TASK.VLN_ORACLE_ACTION_SENSOR.NUM_WAYPOINTS = 0
# -----------------------------------------------------------------------------
# VLN ORACLE GEODESIC ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_GEODESIC_ACTION_SENSOR = Config()
_C.TASK.VLN_ORACLE_GEODESIC_ACTION_SENSOR.TYPE = "VLNOracleActionGeodesicSensor"
_C.TASK.VLN_ORACLE_GEODESIC_ACTION_SENSOR.GOAL_RADIUS = 0.5
_C.TASK.VLN_ORACLE_GEODESIC_ACTION_SENSOR.SPLIT = "train"
_C.TASK.VLN_ORACLE_GEODESIC_ACTION_SENSOR.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz"
)
_C.TASK.VLN_ORACLE_GEODESIC_ACTION_SENSOR.IS_SPARSE = True
_C.TASK.VLN_ORACLE_GEODESIC_ACTION_SENSOR.NUM_WAYPOINTS = 0
# -----------------------------------------------------------------------------
# VLN ORACLE SPLINE ACTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_SPLINE_ACTION_SENSOR = Config()
_C.TASK.VLN_ORACLE_SPLINE_ACTION_SENSOR.TYPE = "VLNOracleActionSplineSensor"
_C.TASK.VLN_ORACLE_SPLINE_ACTION_SENSOR.GOAL_RADIUS = 0.5
_C.TASK.VLN_ORACLE_SPLINE_ACTION_SENSOR.SPLIT = "train"
_C.TASK.VLN_ORACLE_SPLINE_ACTION_SENSOR.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz"
)
_C.TASK.VLN_ORACLE_SPLINE_ACTION_SENSOR.IS_SPARSE = True
_C.TASK.VLN_ORACLE_SPLINE_ACTION_SENSOR.NUM_WAYPOINTS = 1
# -----------------------------------------------------------------------------
# VLN ORACLE PROGRESS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR = Config()
_C.TASK.VLN_ORACLE_PROGRESS_SENSOR.TYPE = "VLNOracleProgressSensor"
# -----------------------------------------------------------------------------
# NDTW MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.NDTW = Config()
_C.TASK.NDTW.TYPE = "NDTW"
_C.TASK.NDTW.SPLIT = "val_seen"
_C.TASK.NDTW.FDTW = True  # False: DTW
_C.TASK.NDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.NDTW.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# SDTW MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SDTW = Config()
_C.TASK.SDTW.TYPE = "SDTW"
_C.TASK.SDTW.SPLIT = "val_seen"
_C.TASK.SDTW.FDTW = True  # False: DTW
_C.TASK.SDTW.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.SDTW.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# Waypoints Accuracy MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.WAYPOINT_ACCURACY = Config()
_C.TASK.WAYPOINT_ACCURACY.TYPE = "WaypointAccuracy"
_C.TASK.WAYPOINT_ACCURACY.SPLIT = "val_seen"
_C.TASK.WAYPOINT_ACCURACY.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json"
)
_C.TASK.WAYPOINT_ACCURACY.SUCCESS_DISTANCE = 0.5
# -----------------------------------------------------------------------------
# PATH_LENGTH MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PATH_LENGTH = Config()
_C.TASK.PATH_LENGTH.TYPE = "PathLength"
# -----------------------------------------------------------------------------
# ORACLE_NAVIGATION_ERROR MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_NAVIGATION_ERROR = Config()
_C.TASK.ORACLE_NAVIGATION_ERROR.TYPE = "OracleNavigationError"
# -----------------------------------------------------------------------------
# ORACLE_SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SUCCESS = Config()
_C.TASK.ORACLE_SUCCESS.TYPE = "OracleSuccess"
_C.TASK.ORACLE_SUCCESS.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# ORACLE_SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SPL = Config()
_C.TASK.ORACLE_SPL.TYPE = "OracleSPL"
_C.TASK.ORACLE_SPL.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# STEPS_TAKEN MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.STEPS_TAKEN = Config()
_C.TASK.STEPS_TAKEN.TYPE = "StepsTaken"
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 3000
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 90
_C.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = False
_C.TASK.TOP_DOWN_MAP.DRAW_WAYPOINTS = True
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE_AND_TARGET = True
# Axes aligned bounding boxes
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = True
_C.TASK.TOP_DOWN_MAP.GT_PATH = (
    "data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz"
)
_C.TASK.TOP_DOWN_MAP.SUCCESS_DISTANCE = 1.5
_C.TASK.TOP_DOWN_MAP.SPLIT = "val_seen"

def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None, opts: Optional[list] = None
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()

    if config_paths:
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)
    config.freeze()
    return config