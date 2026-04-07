"""
Synthetic Motion Video Renderer
================================
A Blender-based pipeline for generating synthetic human motion videos
with automatically extracted 2D pose annotations (COCO-17 keypoints).

This script implements a complete synthetic data generation (SDG) workflow:

    1. Asset Loading    — Import rigged FBX characters with motion-capture animations
    2. Scene Assembly   — Configure virtual camera and HDR environment lighting
    3. Domain Pairing   — Combinatorially pair each character/motion with each background
    4. Rendering        — Produce video frames via the Cycles path-tracing engine
    5. Annotation       — Project 3D armature bones to 2D image coordinates and export
                          per-frame keypoint annotations in a COCO-compatible layout

The core idea behind this pipeline is *domain randomization through combinatorial
scene construction*: by systematically varying the character appearance, motion clip,
and environment lighting, we generate diverse training data whose visual distribution
is broad enough that models trained on it generalize to real-world footage.

Each rendered scene produces:
    - video.mp4        — rendered motion video
    - joints.json      — per-frame COCO-17 keypoints (normalized coordinates)
    - pose.json        — per-frame raw bone projections (pixel coordinates)
    - camera.json      — virtual camera intrinsics and extrinsics

Usage
-----
    blender --background --python render_motion.py -- \\
        --fbx_dir  ./motions \\
        --hdr_dir  ./backgrounds \\
        --out_dir  ./output \\
        --frames   100 \\
        --fps      30

    (Everything after '--' is forwarded to this script by Blender.)

Asset Sources
-------------
    - Character FBX files : https://www.mixamo.com
    - HDR environment maps: https://polyhaven.com  (search "outdoor")
"""

import bpy
import os
import sys
import json
import math
import argparse
import mathutils
from math import radians
from itertools import product
from bpy_extras.object_utils import world_to_camera_view


# ---------------------------------------------------------------------------
# CLI argument parsing (Blender passes script args after '--')
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command-line arguments passed after Blender's '--' separator."""
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(
        description="Render synthetic motion videos with 2D pose annotations."
    )
    parser.add_argument(
        "--fbx_dir", type=str, required=True,
        help="Directory containing FBX character files with skeletal animations."
    )
    parser.add_argument(
        "--hdr_dir", type=str, required=True,
        help="Directory containing HDR/EXR environment maps for scene lighting."
    )
    parser.add_argument(
        "--out_dir", type=str, required=True,
        help="Root output directory. Each (character, background) pair gets a subfolder."
    )
    parser.add_argument(
        "--width", type=int, default=1920,
        help="Render resolution width  (default: 1920)."
    )
    parser.add_argument(
        "--height", type=int, default=1080,
        help="Render resolution height (default: 1080)."
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Output video frame rate (default: 30)."
    )
    parser.add_argument(
        "--frames", type=int, default=100,
        help="Total number of frames to render. Animations are looped via NLA "
             "strips if the source clip is shorter than this value (default: 100)."
    )
    parser.add_argument(
        "--samples", type=int, default=256,
        help="Cycles path-tracing samples per pixel (default: 256)."
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Small normalized padding added to bounding boxes computed from keypoints
BBOX_MARGIN = 0.01

# COCO-17 keypoint names in canonical order.
# This ordering is the standard for human pose estimation benchmarks.
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


# ---------------------------------------------------------------------------
# Rig-to-COCO bone mapping tables
# ---------------------------------------------------------------------------
# Different character rigs name their bones differently.  We maintain lookup
# tables that map each COCO keypoint to the corresponding bone name in the
# two most common rig conventions encountered in FBX assets.

MIXAMO_BONE_MAP = {
    "left_shoulder":  "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow":     "LeftForeArm",
    "right_elbow":    "RightForeArm",
    "left_wrist":     "LeftHand",
    "right_wrist":    "RightHand",
    "left_hip":       "LeftUpLeg",
    "right_hip":      "RightUpLeg",
    "left_knee":      "LeftLeg",
    "right_knee":     "RightLeg",
    "left_ankle":     "LeftFoot",
    "right_ankle":    "RightFoot",
    "head":           "Head",
}

UE_MANNEQUIN_BONE_MAP = {
    "left_shoulder":  "upperarm_l",
    "right_shoulder": "upperarm_r",
    "left_elbow":     "lowerarm_l",
    "right_elbow":    "lowerarm_r",
    "left_wrist":     "hand_l",
    "right_wrist":    "hand_r",
    "left_hip":       "thigh_l",
    "right_hip":      "thigh_r",
    "left_knee":      "calf_l",
    "right_knee":     "calf_r",
    "left_ankle":     "foot_l",
    "right_ankle":    "foot_r",
    "head":           "head",
}


# ---------------------------------------------------------------------------
# Scene management
# ---------------------------------------------------------------------------

def clear_scene():
    """Remove every object, action, and armature from the current Blender scene.

    This ensures a clean slate before each render so that assets from a
    previous combination do not leak into the next one.
    """
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in list(bpy.data.actions):
        bpy.data.actions.remove(block)
    for block in list(bpy.data.armatures):
        bpy.data.armatures.remove(block)


def import_fbx(fbx_path, frame_start, frame_end):
    """Import an FBX character and configure its animation for looped playback.

    Many motion-capture clips are shorter than the desired render length.
    To handle this, we:
      1. Add a CYCLES modifier to every F-Curve so Blender extrapolates
         the animation beyond its native range.
      2. Push the action into an NLA strip with an explicit repeat count,
         ensuring seamless looping over the full frame range.

    Args:
        fbx_path:    Path to the FBX file.
        frame_start: First frame of the output sequence.
        frame_end:   Last frame of the output sequence.
    """
    bpy.ops.import_scene.fbx(filepath=fbx_path)

    # Reset imported objects to the world origin
    for obj in bpy.context.selected_objects:
        if obj.type in ('ARMATURE', 'MESH'):
            obj.location = (0, 0, 0)

    bpy.context.view_layer.update()

    # --- Loop animation via F-Curve modifiers ---
    for action in bpy.data.actions:
        for fcu in action.fcurves:
            if not any(m.type == 'CYCLES' for m in fcu.modifiers):
                fcu.modifiers.new(type='CYCLES')

    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end   = frame_end

    # --- Push actions into NLA strips with explicit repeat counts ---
    needed_length = frame_end - frame_start + 1
    for obj in bpy.context.selected_objects:
        anim_data = obj.animation_data
        if anim_data and anim_data.action:
            action = anim_data.action
            anim_data.use_nla = True

            track = anim_data.nla_tracks.new()
            strip = track.strips.new(
                name="LoopStrip", start=frame_start, action=action
            )

            clip_start, clip_end = action.frame_range
            clip_length = max(1, int(round(clip_end - clip_start)))

            strip.action_frame_start = clip_start
            strip.action_frame_end   = clip_end
            strip.repeat    = math.ceil(needed_length / clip_length)
            strip.frame_end = frame_start + strip.repeat * (clip_end - clip_start)

            # Let NLA drive the animation (detach the direct action)
            anim_data.action = None

    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

def create_camera():
    """Create and position a virtual camera for a front-facing view.

    The camera is placed at a fixed distance along the negative Y-axis,
    looking toward the origin.  This simulates a typical tripod setup for
    capturing full-body human motion.

    Returns:
        The Blender camera object.
    """
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj  = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_obj.location       = (0.0, -8.0, 1.0)
    cam_obj.rotation_euler = (radians(90), 0, radians(0))

    return cam_obj


def extract_camera_info(camera):
    """Serialize camera intrinsics and extrinsics to a plain dictionary.

    This information is sufficient to reproduce the exact 3D-to-2D
    projection used during rendering (pinhole camera model).
    """
    cam   = camera.data
    scene = bpy.context.scene
    return {
        "location":       [float(v) for v in camera.location],
        "rotation_euler": [float(v) for v in camera.rotation_euler],
        "lens":           float(cam.lens),
        "sensor_width":   float(cam.sensor_width),
        "sensor_height":  float(cam.sensor_height),
        "resolution":     [scene.render.resolution_x, scene.render.resolution_y],
        "clip_start":     float(cam.clip_start),
        "clip_end":       float(cam.clip_end),
    }


# ---------------------------------------------------------------------------
# Environment lighting (HDR background)
# ---------------------------------------------------------------------------

def setup_hdri_background(hdr_path):
    """Load an HDR environment map and wire it into the world shader.

    HDR environment maps provide realistic, 360-degree lighting that
    wraps around the entire scene.  Using diverse outdoor HDRIs from
    sources like Poly Haven introduces natural variation in illumination
    color, direction, and intensity — a simple but effective form of
    domain randomization.
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    tree = world.node_tree
    tree.nodes.clear()

    env_node = tree.nodes.new('ShaderNodeTexEnvironment')
    env_node.image = bpy.data.images.load(hdr_path)

    bg_node = tree.nodes.new('ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = 2.0

    out_node = tree.nodes.new('ShaderNodeOutputWorld')

    tree.links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
    tree.links.new(bg_node.outputs['Background'], out_node.inputs['Surface'])


# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------

def configure_renderer(out_path, resolution, fps, samples, frame_start, frame_end):
    """Configure the Cycles renderer for video output.

    We use Cycles (a physically-based path tracer) rather than EEVEE
    because path tracing produces more realistic lighting, shadows, and
    material interactions — important for minimizing the sim-to-real gap
    in the generated training data.

    GPU rendering via CUDA is preferred when available; the script falls
    back to CPU rendering otherwise.
    """
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    # Attempt GPU acceleration
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        scene.cycles.device = 'GPU'
    except Exception:
        scene.cycles.device = 'CPU'

    scene.cycles.samples               = samples
    scene.cycles.max_bounces           = 6
    scene.cycles.use_denoising         = False
    scene.render.resolution_percentage = 100
    scene.render.resolution_x          = resolution[0]
    scene.render.resolution_y          = resolution[1]
    scene.render.fps                   = fps
    scene.frame_start                  = frame_start
    scene.frame_end                    = frame_end

    # Output as H.264 MP4
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format  = 'MPEG4'
    scene.render.ffmpeg.codec   = 'H264'
    scene.render.filepath       = out_path


# ---------------------------------------------------------------------------
# Rig detection and bone resolution
# ---------------------------------------------------------------------------

def detect_rig_type(armature_obj):
    """Auto-detect which bone-naming convention the armature uses.

    Mixamo characters use CamelCase names (e.g., 'LeftForeArm'),
    while Unreal Engine mannequins use snake_case (e.g., 'lowerarm_l').
    We check for distinctive bone names to pick the right mapping table.
    """
    bone_names = {b.name.lower() for b in armature_obj.data.bones}
    if any("leftforearm" in n for n in bone_names) or \
       any("mixamorig" in n for n in bone_names):
        return MIXAMO_BONE_MAP
    if any("lowerarm_l" in n for n in bone_names):
        return UE_MANNEQUIN_BONE_MAP
    # Default to Mixamo (most common for downloaded FBX assets)
    return MIXAMO_BONE_MAP


def _resolve_bone(armature_obj, target_name):
    """Fuzzy-match a target bone name against the armature's pose bones.

    FBX exporters sometimes prepend a namespace prefix (e.g.,
    'mixamorig:LeftArm') or use inconsistent casing.  This function
    tries exact match first, then progressively looser matching.
    """
    if not target_name:
        return None
    pose_bones = armature_obj.pose.bones
    # Exact match
    if target_name in pose_bones:
        return target_name
    t = target_name.lower()
    # Case-insensitive exact match
    for name in pose_bones.keys():
        if name.lower() == t:
            return name
    # Suffix match (handles namespace prefixes)
    for name in pose_bones.keys():
        if name.lower().endswith(t):
            return name
    # Substring match (last resort)
    for name in pose_bones.keys():
        if t in name.lower():
            return name
    return None


# ---------------------------------------------------------------------------
# 3D-to-2D keypoint projection
# ---------------------------------------------------------------------------

def _bone_world_position(armature_obj, bone_name):
    """Get the world-space position of a bone's head joint."""
    resolved = _resolve_bone(armature_obj, bone_name)
    if resolved is None:
        return None
    return armature_obj.matrix_world @ armature_obj.pose.bones[resolved].head


def _approximate_face_keypoints(armature_obj, head_world, head_bone_name):
    """Estimate face keypoint positions from the head bone's orientation.

    Since character rigs typically have a single 'Head' bone without
    individual facial landmarks, we approximate COCO face keypoints
    (nose, eyes, ears) using fixed offsets along the head bone's local
    forward and lateral axes.

    NOTE: These are geometric approximations, NOT true facial landmarks.
    For applications requiring precise face keypoints, a dedicated facial
    rig or a post-hoc face detector should be used instead.
    """
    try:
        resolved = _resolve_bone(armature_obj, head_bone_name)
        head_pb  = armature_obj.pose.bones[resolved] if resolved else None
        head_mtx = (armature_obj.matrix_world @ head_pb.matrix
                     if head_pb else mathutils.Matrix.Identity(4))
        rot = head_mtx.to_3x3()
        forward = (rot @ mathutils.Vector((0, 0, -1))).normalized()
        right   = (rot @ mathutils.Vector((1, 0, 0))).normalized()
    except Exception:
        forward = mathutils.Vector((0, 0, -1))
        right   = mathutils.Vector((1, 0, 0))

    return {
        "nose":      head_world + 0.05 * forward,
        "left_eye":  head_world + 0.03 * forward - 0.03 * right,
        "right_eye": head_world + 0.03 * forward + 0.03 * right,
        "left_ear":  head_world - 0.02 * right,
        "right_ear": head_world + 0.02 * right,
    }


def _project_to_image(scene, camera_obj, world_coord, clamp=True):
    """Project a 3D world coordinate to normalized 2D image coordinates.

    Returns:
        (x, y, visibility) where x,y are in [0,1] (top-left origin) and
        visibility is 2 (visible) or 0 (outside frame / behind camera).
    """
    if world_coord is None:
        return 0.0, 0.0, 0
    ndc = world_to_camera_view(scene, camera_obj, world_coord)
    x, y_ndc, z = ndc
    y = 1.0 - y_ndc  # Convert to top-left origin

    # Visible if in front of camera and within frame bounds
    visible = 2 if (z >= 0.0 and 0.0 <= ndc.x <= 1.0 and 0.0 <= y_ndc <= 1.0) else 0

    if clamp:
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
    return float(x), float(y), int(visible)


def _extract_keypoints_for_frame(armature_obj, camera_obj, scene, bone_map):
    """Extract normalized COCO-17 keypoints for one frame.

    Body keypoints (shoulders → ankles) are exact projections of the
    corresponding armature bone heads.  Face keypoints are approximated
    from the head bone orientation (see _approximate_face_keypoints).
    """
    world_positions = {k: None for k in COCO_KEYPOINT_NAMES}

    # Body joints — direct bone-head projection
    body_joints = [
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    for joint in body_joints:
        world_positions[joint] = _bone_world_position(
            armature_obj, bone_map.get(joint, "")
        )

    # Face keypoints — approximated from head bone
    head_bone  = bone_map.get("head", "")
    head_world = _bone_world_position(armature_obj, head_bone)
    if head_world is not None:
        face_pts = _approximate_face_keypoints(
            armature_obj, head_world, head_bone
        )
        for name in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            world_positions[name] = face_pts[name]

    # Project all keypoints to 2D
    keypoints, visibility = [], []
    for name in COCO_KEYPOINT_NAMES:
        x, y, v = _project_to_image(scene, camera_obj, world_positions[name])
        keypoints.append([x, y])
        visibility.append(v)
    return keypoints, visibility


def _compute_bbox(keypoints, visibility):
    """Compute a bounding box from visible keypoints (normalized coords).

    The bbox is returned in [x, y, w, h] format with a small margin.
    """
    visible_idxs = [i for i, v in enumerate(visibility) if v == 2]
    if not visible_idxs:
        mid = 0.5 - BBOX_MARGIN
        return [mid, mid, 2 * BBOX_MARGIN, 2 * BBOX_MARGIN]

    xs = [keypoints[i][0] for i in visible_idxs]
    ys = [keypoints[i][1] for i in visible_idxs]
    x_min = max(0.0, min(xs) - BBOX_MARGIN)
    y_min = max(0.0, min(ys) - BBOX_MARGIN)
    x_max = min(1.0, max(xs) + BBOX_MARGIN)
    y_max = min(1.0, max(ys) + BBOX_MARGIN)
    return [x_min, y_min, max(0.0, x_max - x_min), max(0.0, y_max - y_min)]


# ---------------------------------------------------------------------------
# Raw bone projection (pixel coordinates, all bones)
# ---------------------------------------------------------------------------

def get_all_bone_projections(armature_obj, camera_obj, resolution):
    """Project every bone head to pixel coordinates (for pose.json).

    Unlike the COCO-17 extraction above, this captures ALL bones in the
    armature using their original rig names and absolute pixel positions.
    """
    scene = bpy.context.scene
    coords = {}
    for bone in armature_obj.pose.bones:
        world_co = armature_obj.matrix_world @ bone.head
        ndc = world_to_camera_view(scene, camera_obj, world_co)
        coords[bone.name] = [
            int(ndc.x * resolution[0]),
            int((1 - ndc.y) * resolution[1]),
        ]
    return coords


# ---------------------------------------------------------------------------
# Annotation export
# ---------------------------------------------------------------------------

def export_joints_json(out_dir, armatures, camera_obj, frame_start, frame_end):
    """Export per-frame COCO-17 keypoints to joints.json.

    Output format (per frame):
        {
            "frame_id": int,
            "instances": [
                {
                    "boxes":      [x, y, w, h],        // normalized [0,1]
                    "bbox_score": 1.0,
                    "keypoints":  [[x, y], ...] x 17    // normalized [0,1]
                }
            ]
        }

    IMPORTANT: These keypoints are 3D-to-2D bone projections (synthetic
    ground truth), NOT predictions from a pose estimator.  Coordinates are
    normalized to [0,1] with top-left origin.
    """
    scene = bpy.context.scene
    bone_maps = {arm.name: detect_rig_type(arm) for arm in armatures}
    frames = []

    for idx, frame in enumerate(range(frame_start, frame_end + 1)):
        scene.frame_set(frame)
        instances = []
        for arm in armatures:
            kps, vis = _extract_keypoints_for_frame(
                arm, camera_obj, scene, bone_maps[arm.name]
            )
            bbox = _compute_bbox(kps, vis)
            instances.append({
                "boxes": bbox,
                "bbox_score": 1.0,
                "keypoints": kps,
            })
        frames.append({"frame_id": idx, "instances": instances})

    out_path = os.path.join(out_dir, "joints.json")
    with open(out_path, "w") as f:
        json.dump(frames, f, indent=2)
    print(f"[OK] joints.json -> {out_path}")


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def render_combination(fbx_path, hdr_path, cfg):
    """Render one (character, background) combination and export all annotations."""
    base_fbx = os.path.splitext(os.path.basename(fbx_path))[0]
    base_hdr = os.path.splitext(os.path.basename(hdr_path))[0]
    out_dir  = os.path.join(cfg.out_dir, f"{base_fbx}_{base_hdr}")

    # Skip if already rendered (enables safe re-runs)
    if os.path.exists(out_dir):
        print(f"[SKIP] Already exists: {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)

    video_path  = os.path.join(out_dir, "video.mp4")
    resolution  = (cfg.width, cfg.height)
    frame_start = 1
    frame_end   = cfg.frames

    # --- Build scene ---
    clear_scene()
    import_fbx(fbx_path, frame_start, frame_end)
    camera = create_camera()
    setup_hdri_background(hdr_path)

    # --- Render video ---
    configure_renderer(video_path, resolution, cfg.fps, cfg.samples,
                       frame_start, frame_end)
    bpy.ops.render.render(animation=True)

    # --- Export raw bone projections (pose.json) ---
    armature = next(
        obj for obj in bpy.context.scene.objects if obj.type == 'ARMATURE'
    )
    poses = {}
    for f in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(f)
        poses[f"frame_{f:04d}"] = get_all_bone_projections(
            armature, camera, resolution
        )
    with open(os.path.join(out_dir, "pose.json"), "w") as fp:
        json.dump(poses, fp, indent=2)

    # --- Export camera parameters (camera.json) ---
    with open(os.path.join(out_dir, "camera.json"), "w") as fc:
        json.dump(extract_camera_info(camera), fc, indent=2)

    # --- Export COCO-17 keypoints (joints.json) ---
    all_armatures = [
        o for o in bpy.context.scene.objects if o.type == 'ARMATURE'
    ]
    export_joints_json(out_dir, all_armatures, camera, frame_start, frame_end)

    print(f"[DONE] {base_fbx} + {base_hdr} -> {out_dir}")


def main():
    cfg = parse_args()

    # Discover all FBX files
    fbx_files = sorted([
        os.path.join(cfg.fbx_dir, f)
        for f in os.listdir(cfg.fbx_dir)
        if f.lower().endswith(".fbx")
    ])

    # Discover all HDR/EXR environment maps
    hdr_files = sorted([
        os.path.join(cfg.hdr_dir, f)
        for f in os.listdir(cfg.hdr_dir)
        if f.lower().endswith((".hdr", ".exr"))
    ])

    if not fbx_files:
        print(f"[ERROR] No .fbx files found in {cfg.fbx_dir}")
        return
    if not hdr_files:
        print(f"[ERROR] No .hdr/.exr files found in {cfg.hdr_dir}")
        return

    # Combinatorial rendering: every character x every background
    combos = list(product(fbx_files, hdr_files))
    total  = len(combos)
    print(f"[INFO] {len(fbx_files)} characters x {len(hdr_files)} backgrounds = {total} combinations")

    for i, (fbx, hdr) in enumerate(combos, start=1):
        print(f"\n[{i}/{total}] Rendering: {os.path.basename(fbx)} + {os.path.basename(hdr)}")
        try:
            render_combination(fbx, hdr, cfg)
        except Exception as e:
            print(f"[ERROR] {os.path.basename(fbx)} + {os.path.basename(hdr)}: {e}")


if __name__ == "__main__":
    main()
