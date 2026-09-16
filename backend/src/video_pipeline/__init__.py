"""Video pipeline package — extensible, stage-based video processing."""
from video_pipeline.context import VideoContext, VideoMetadata
from video_pipeline.camera_motion import CameraMotionStage, MotionConfig
from video_pipeline.keyframes import KeyframeConfig, KeyframeSelectionStage
from video_pipeline.local_map import LocalMapConfig, LocalMappingStage, MAP_SCALE_NOTE
from video_pipeline.optimization import (
    OPT_SCALE_NOTE, OptimizationConfig, PoseOptimizationStage,
    build_observations, project_points, reprojection_residuals, rms)
from video_pipeline.orb_features import OrbConfig, OrbFeatureStage
from video_pipeline.orb_matching import DescriptorMatchingStage, MatchConfig
from video_pipeline.pipeline import VideoPipeline
from video_pipeline.point_filter import FILTER_SCALE_NOTE, PointFilterConfig, PointFilterStage
from video_pipeline.stages import FrameSamplingStage, ProbeStage, Stage
from video_pipeline.trajectory import TrajectoryConfig, TrajectoryStage
from video_pipeline.triangulation import POINTS_SCALE_NOTE, TriangulationConfig, TriangulationStage

__all__ = ["VideoContext", "VideoMetadata", "VideoPipeline", "Stage", "ProbeStage", "FrameSamplingStage",
           "OrbConfig", "OrbFeatureStage", "MatchConfig", "DescriptorMatchingStage",
           "MotionConfig", "CameraMotionStage", "TrajectoryConfig", "TrajectoryStage",
           "TriangulationConfig", "TriangulationStage", "POINTS_SCALE_NOTE",
           "PointFilterConfig", "PointFilterStage", "FILTER_SCALE_NOTE",
           "KeyframeConfig", "KeyframeSelectionStage",
           "LocalMapConfig", "LocalMappingStage", "MAP_SCALE_NOTE",
           "OptimizationConfig", "PoseOptimizationStage", "OPT_SCALE_NOTE",
           "build_observations", "project_points", "reprojection_residuals", "rms"]
