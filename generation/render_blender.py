import bpy
import math
import numpy as np
import csv
import os
import json
import subprocess

# ==========================================
# 0. CONFIGURATION — CHANGE THESE BEFORE EACH RUN
# ==========================================
# Select which RPM/slip condition to simulate.
# These can be overridden without editing the file, e.g.:
#   OMEGA_P_RPM=78 SC_TARGET=0.95 blender --background --python shell_simulation_blender.py
OMEGA_P_RPM = int(os.environ.get("OMEGA_P_RPM", 78))

# Head-contact slip at C under the no-slip-at-A/B model.
#   S_C = 0 -> pure rolling everywhere
#   S_C = 1 -> shell center moves with the groove, local shell spin goes to zero,
#              and the idle top pad sees pure sliding.
SC_TARGET = max(0.0, min(1.0, float(os.environ.get("SC_TARGET", 0.95))))
SC_VARIATION = float(os.environ.get("SC_VARIATION", 0.05))
RUN_GROUP = os.environ.get("RUN_GROUP", "controlled_c_slip")
RUN_INDEX = os.environ.get("RUN_INDEX", "single")

# Batch driver: run this script once and it launches one background Blender
# render per S_C value. Set BATCH_RENDER=0 for a single interactive scene.
BATCH_RENDER = os.environ.get("BATCH_RENDER", "1") == "1"
AUTO_RENDER = os.environ.get("AUTO_RENDER", "0") == "1"
BATCH_RPM_VALUES = [
    int(x.strip()) for x in os.environ.get(
        "BATCH_RPM_VALUES",
        "21,48,78,108",
    ).split(",")
    if x.strip()
]
BATCH_SC_VALUES = [
    float(x.strip()) for x in os.environ.get(
        "BATCH_SC_VALUES",
        "0.10,0.35,0.50,0.65,0.85",
    ).split(",")
    if x.strip()
]

# All runs use these fixed parameters
NUM_SHELLS       = 10
FPS              = 100
TOTAL_FRAMES     = int(os.environ.get("TOTAL_FRAMES", 800))  # exactly 8 s at 100 FPS
DURATION_SEC     = TOTAL_FRAMES / FPS
SEED             = int(os.environ.get("SIMULATION_SEED", 42))
MOTION_BLUR_SHUTTER = 0.0       # no motion blur
LIGHTING_ENERGY  = 8000
WOBBLE_SCALE     = 0.5

# Backward-compatible metadata key used by older analysis scripts.
SC_MEAN_BASE     = SC_TARGET

# ==========================================
# 1. PHYSICS PARAMETERS
# ==========================================
Rb = 0.01          # Radius of the shell (m)
Rg = 0.3168        # Radius of the V-groove (m)
theta = 45         # Half angle of V-groove (degrees)

omega_p = OMEGA_P_RPM * 2 * np.pi / 60  # rad/sec
theta_rad = np.radians(theta)

# Contact radii
Ri = Rg - Rb * np.cos(theta_rad)   # inner ball-groove contact radius
Ro = Rg + Rb * np.cos(theta_rad)   # outer ball-groove contact radius

# Pure-rolling reference kinematics: S_C = 0
omega_c = omega_p / (1 + np.sin(theta_rad))
beta    = np.arctan((Rb * np.sin(theta_rad)) / Rg)
omega_b = (omega_p * np.sqrt(Rg**2 + Rb**2 * np.sin(theta_rad)**2)
           / (Rb * (1 + np.sin(theta_rad))))

# Target no-slip-at-A/B, slip-at-C kinematics.
omega_c_target = (omega_p / (1 + np.sin(theta_rad))) * (1 + SC_TARGET * np.sin(theta_rad))
omega_b_target = (Rg * omega_p * (1 - SC_TARGET)) / (Rb * np.cos(beta) * (1 + np.sin(theta_rad)))

# ---- Output directory ----
_blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else os.getcwd()
OUTPUT_ROOT = os.environ.get(
    "OUTPUT_ROOT",
    os.path.join(_blend_dir, "synthetic_polishing_videos"),
)

def _resolve_batch_script_path():
    """Find a real script path even when run from Blender's Text Editor."""
    candidates = []

    env_path = os.environ.get("CSLIP_SCRIPT_PATH", "")
    if env_path:
        candidates.append(env_path)

    if "__file__" in globals():
        candidates.append(__file__)

    try:
        text = getattr(getattr(bpy.context, "space_data", None), "text", None)
        if text and text.filepath:
            candidates.append(bpy.path.abspath(text.filepath))
    except Exception:
        pass

    for text in bpy.data.texts:
        try:
            if text.filepath:
                candidates.append(bpy.path.abspath(text.filepath))
        except Exception:
            pass

    candidates.extend([
        os.path.join(os.getcwd(), "blender_scripts", "shell_simulation_blender.py"),
        os.path.join(_blend_dir, "blender_scripts", "shell_simulation_blender.py"),
    ])

    for path in candidates:
        path = os.path.abspath(path)
        if os.path.exists(path):
            return path

    # Last resort for "Run Script" from an unsaved Text Editor block.
    try:
        text = getattr(getattr(bpy.context, "space_data", None), "text", None)
        if text is None:
            text = next((t for t in bpy.data.texts if "shell_simulation" in t.name), None)
        if text is not None:
            tmp_dir = os.path.join("/tmp", "controlled_c_slip_blender")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, "shell_simulation_blender_batch.py")
            with open(tmp_path, "w") as f:
                f.write(text.as_string())
            return tmp_path
    except Exception:
        pass

    return None

if BATCH_RENDER and os.environ.get("_CSLIP_BATCH_CHILD") != "1":
    script_path = _resolve_batch_script_path()
    if not script_path:
        raise RuntimeError(
            "Batch render could not resolve this script path. Save the Text Editor "
            "script or run: blender --background --python blender_scripts/shell_simulation_blender.py"
        )

    print("\nBATCH_RENDER=1: starting controlled C-slip batch launcher.")
    manifest_dir = os.path.join(OUTPUT_ROOT, RUN_GROUP)
    os.makedirs(manifest_dir, exist_ok=True)
    manifest = {
        "description": "Controlled C-slip sweep; no slip at A/B, variable slip at C",
        "regime": "controlled_c_slip",
        "slip_type": "S_C",
        "contact_constraints": "V_A=V_B=0; V_C=S_C*R_g*omega_p",
        "rpm_values": BATCH_RPM_VALUES,
        "fps": FPS,
        "total_frames": TOTAL_FRAMES,
        "sc_values": BATCH_SC_VALUES,
        "run_group": RUN_GROUP,
        "batch_script_path": script_path,
    }
    manifest_path = os.path.join(manifest_dir, "batch_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_jobs = len(BATCH_RPM_VALUES) * len(BATCH_SC_VALUES)
    print(f"\nLaunching batch render: {total_jobs} videos")
    print(f"Script: {script_path}")
    print(f"Manifest: {manifest_path}")
    job_idx = 0
    for rpm_value in BATCH_RPM_VALUES:
        for sc_value in BATCH_SC_VALUES:
            env = os.environ.copy()
            env.update({
                "_CSLIP_BATCH_CHILD": "1",
                "BATCH_RENDER": "0",
                "AUTO_RENDER": "1",
                "OMEGA_P_RPM": str(rpm_value),
                "SC_TARGET": f"{sc_value:.6f}",
                "SC_VARIATION": f"{SC_VARIATION:.6f}",
                "RUN_GROUP": RUN_GROUP,
                "RUN_INDEX": f"{job_idx:03d}",
                "TOTAL_FRAMES": str(TOTAL_FRAMES),
                "OUTPUT_ROOT": OUTPUT_ROOT,
            })
            cmd = [bpy.app.binary_path, "--background", "--python", script_path]
            print(
                f"[{job_idx + 1:02d}/{total_jobs:02d}] "
                f"RPM={rpm_value} S_C={sc_value:.2f}"
            )
            subprocess.run(cmd, env=env, check=True)
            job_idx += 1
    print("Batch render complete.")
    raise SystemExit

SC_LABEL = f"sc{SC_TARGET:.2f}".replace(".", "p")
RPM_LABEL = f"rpm_{OMEGA_P_RPM:03d}"
RUN_ID = f"seq_{RUN_INDEX}_rpm{OMEGA_P_RPM:03d}_{SC_LABEL}"
OUTPUT_DIR  = os.path.join(OUTPUT_ROOT, RUN_GROUP, RPM_LABEL, RUN_ID)
if os.path.exists(OUTPUT_DIR):
    raise FileExistsError("Refusing to overwrite an existing simulation")
os.makedirs(OUTPUT_DIR, exist_ok=False)
os.makedirs(os.path.join(OUTPUT_DIR, "frames"), exist_ok=True)

print(f"\n{'='*50}")
print(f"C-SLIP SWEEP — {OMEGA_P_RPM} RPM, S_C={SC_TARGET:.3f}")
print(f"  Model: no slip at A/B, slip only at C (idle top pad)")
print(f"  S_C=0 pure rolling; S_C=1 pure sliding at C")
print(f"  Frames: {TOTAL_FRAMES}  |  Duration: {DURATION_SEC}s  |  FPS: {FPS}")
print(f"  Output: {OUTPUT_DIR}")
print(f"{'='*50}\n")

print(f"Reference pure-rolling kinematics (S_C=0):")
print(f"  Ri = {Ri*1000:.3f} mm  |  Ro = {Ro*1000:.3f} mm")
print(f"  omega_c = {omega_c:.4f} rad/s")
print(f"  beta    = {np.degrees(beta):.4f} deg")
print(f"  omega_b = {omega_b:.4f} rad/s")
print(f"Target kinematics:")
print(f"  omega_c_target = {omega_c_target:.4f} rad/s")
print(f"  omega_b_target = {omega_b_target:.4f} rad/s")

# ==========================================
# 2. BLENDER SCENE SETUP
# ==========================================

# Clean Slate
if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

scene = bpy.context.scene
scene.render.fps = FPS
scene.frame_start = 0
scene.frame_end = TOTAL_FRAMES - 1

RENDER_W = 1920
RENDER_H = 1080
scene.render.resolution_x = RENDER_W
scene.render.resolution_y = RENDER_H

# Motion Blur
if MOTION_BLUR_SHUTTER > 0:
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = MOTION_BLUR_SHUTTER
else:
    scene.render.use_motion_blur = False

# Render Engine
scene.render.engine = 'BLENDER_EEVEE'

try:
    if hasattr(scene, "eevee"):
        scene.eevee.use_bloom = True
        scene.eevee.bloom_intensity = 0.22
        scene.eevee.bloom_radius = 7.0
        scene.eevee.bloom_threshold = 0.40
        if hasattr(scene.eevee, "bloom_color"):
            scene.eevee.bloom_color = (1.0, 0.97, 0.60)
except Exception:
    pass

try:
    if hasattr(scene, "eevee"):
        if hasattr(scene.eevee, 'use_ssr'):
            scene.eevee.use_ssr = True
            scene.eevee.use_ssr_refraction = True
            scene.eevee.ssr_quality = 1.0
        if hasattr(scene.eevee, 'use_raytracing'):
            scene.eevee.use_raytracing = True
except Exception:
    pass

try:
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'High Contrast'
    scene.view_settings.exposure = -0.5
    scene.view_settings.gamma = 1.0
except Exception:
    pass

# Collection
coll = bpy.data.collections.new("Dynamics_Simulation")
scene.collection.children.link(coll)

# Orbit controller for shell centers. The physical groove drive is omega_p;
# shell circulation follows omega_c(S_C) from the kinematic model.
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
turntable = bpy.context.active_object
turntable.name = "Shell_Orbit_Controller"

# ==========================================
# 3. MATERIALS
# ==========================================

# Shell material
mat = bpy.data.materials.new(name="Striped_Mat")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

node_out = nodes.new(type='ShaderNodeOutputMaterial')
node_out.location = (400, 0)
node_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
node_bsdf.location = (0, 0)
node_bsdf.inputs['Base Color'].default_value = (0.03, 0.05, 0.03, 1)
node_bsdf.inputs['Metallic'].default_value = 0.15
node_bsdf.inputs['Roughness'].default_value = 0.55
node_bsdf.inputs['Specular IOR Level'].default_value = 0.4
links.new(node_bsdf.outputs[0], node_out.inputs[0])

# Bright asymmetric marker: makes local shell spin observable in rendered
# frames. Without this, low/high-spin cases can look almost identical.
marker_mat = bpy.data.materials.new(name="Shell_Spin_Marker_Mat")
marker_mat.use_nodes = True
marker_nodes = marker_mat.node_tree.nodes
marker_links = marker_mat.node_tree.links
marker_nodes.clear()
marker_out = marker_nodes.new(type='ShaderNodeOutputMaterial')
marker_bsdf = marker_nodes.new(type='ShaderNodeBsdfPrincipled')
marker_bsdf.inputs['Base Color'].default_value = (0.85, 0.95, 0.35, 1)
marker_bsdf.inputs['Metallic'].default_value = 0.0
marker_bsdf.inputs['Roughness'].default_value = 0.35
if 'Emission Color' in marker_bsdf.inputs:
    marker_bsdf.inputs['Emission Color'].default_value = (0.20, 0.35, 0.05, 1)
if 'Emission Strength' in marker_bsdf.inputs:
    marker_bsdf.inputs['Emission Strength'].default_value = 0.12
marker_links.new(marker_bsdf.outputs[0], marker_out.inputs[0])

# Plate material
plate_mat = bpy.data.materials.new(name="Plate_Brass_Mat")
plate_mat.use_nodes = True
plate_nodes = plate_mat.node_tree.nodes
plate_links = plate_mat.node_tree.links
plate_nodes.clear()

plate_out = plate_nodes.new(type='ShaderNodeOutputMaterial')
plate_out.location = (400, 0)
plate_bsdf = plate_nodes.new(type='ShaderNodeBsdfPrincipled')
plate_bsdf.location = (0, 0)
plate_bsdf.inputs['Base Color'].default_value = (0.15, 0.14, 0.09, 1)
plate_bsdf.inputs['Metallic'].default_value = 0.20
plate_bsdf.inputs['Roughness'].default_value = 0.65
plate_bsdf.inputs['Specular IOR Level'].default_value = 0.15
plate_links.new(plate_bsdf.outputs[0], plate_out.inputs[0])

# ==========================================
# 4. GENERATE GEOMETRY
# ==========================================
angle_step = (2 * np.pi) / NUM_SHELLS

shells = []
for i in range(NUM_SHELLS):
    angle = i * angle_step
    x = Rg * np.cos(angle)
    y = Rg * np.sin(angle)
    z = 0

    bpy.ops.mesh.primitive_uv_sphere_add(radius=Rb, location=(x, y, z))
    sphere = bpy.context.active_object
    sphere.name = f"Shell_{i:03d}"
    sphere.data.materials.append(mat)
    bpy.ops.object.shade_smooth()

    sphere.rotation_euler[2] = angle + (np.pi / 2)
    sphere.parent = turntable

    for c in sphere.users_collection:
        c.objects.unlink(sphere)
    coll.objects.link(sphere)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=Rb * 0.22, location=(0, 0, 0))
    marker = bpy.context.active_object
    marker.name = f"Shell_{i:03d}_SpinMarker"
    marker.data.materials.append(marker_mat)
    bpy.ops.object.shade_smooth()
    marker.parent = sphere
    marker.location = (0.0, 0.0, Rb * 0.82)
    for c in marker.users_collection:
        c.objects.unlink(marker)
    coll.objects.link(marker)

    shells.append(sphere)

plate_radius = Rg + Rb * 2

# Top Plate (IDLE — does not rotate)
bpy.ops.mesh.primitive_cylinder_add(radius=plate_radius, depth=Rb * 2.0,
                                     location=(0, 0, Rb + Rb * 1.0))
top = bpy.context.active_object
top.name = "Top_Plate_IDLE"
top.data.materials.append(plate_mat)
bpy.ops.object.shade_smooth()

# Bottom plate with groove. The visible plate is static; shell motion/GT carry
# the no-slip-at-A/B kinematics.
bottom_plate_height = Rb * 2
groove_cut_height = -Rb * 0.25
bpy.ops.mesh.primitive_cylinder_add(
    radius=plate_radius,
    depth=bottom_plate_height,
    location=(0, 0, groove_cut_height - bottom_plate_height / 2)
)
bot = bpy.context.active_object
bot.name = "Bottom_Plate_GROOVE_VISUAL"
bot.data.materials.append(plate_mat)
bpy.ops.object.shade_smooth()

# Groove cut
bpy.ops.mesh.primitive_torus_add(
    major_radius=Rg,
    minor_radius=Rb * 1.1,
    location=(0, 0, groove_cut_height)
)
groove_cutter = bpy.context.active_object
groove_cutter.name = "Groove_Cutter"

mod = bot.modifiers.new(name="Groove_Boolean", type='BOOLEAN')
mod.operation = 'DIFFERENCE'
mod.object = groove_cutter
bpy.context.view_layer.objects.active = bot
bpy.ops.object.modifier_apply(modifier="Groove_Boolean")
bpy.ops.object.shade_smooth()
bpy.data.objects.remove(groove_cutter, do_unlink=True)

# Bottom plate material variant
bottom_plate_mat = plate_mat.copy()
bottom_plate_mat.name = "Bottom_Plate_Brass_Mat"
bp_nodes = bottom_plate_mat.node_tree.nodes
bp_bsdf = None
for n in bp_nodes:
    if n.type == 'BSDF_PRINCIPLED':
        bp_bsdf = n
        break
if bp_bsdf:
    bp_bsdf.inputs['Base Color'].default_value = (0.27, 0.27, 0.14, 1)
    bp_bsdf.inputs['Roughness'].default_value = 0.72
    bp_bsdf.inputs['Metallic'].default_value = 0.15
    bp_bsdf.inputs['Specular IOR Level'].default_value = 0.12
bot.data.materials.clear()
bot.data.materials.append(bottom_plate_mat)

# ==========================================
# 5. PRE-COMPUTE SLIP, RANDOMIZER & WOBBLE DATA
# ==========================================
# SLIP MODEL: slip at contact point C, with A/B constrained to no slip.
# A high S_C is the groove-attached / idle-top-pad regime: shell spin drops
# and the top contact becomes mostly sliding.
# Contacts A (inner) and B (outer) are pure rolling (V_A = V_B = 0).
# Lee et al. (2006), concentric V-groove (e=0):
#   omega_c = (omega_p / (1+sin(theta))) * (1 + S_C * sin(theta))
#   omega_b = R_g * omega_p * (1 - S_C) / (R_b * cos(beta) * (1 + sin(theta)))
#   tan(beta) = R_b * sin(theta) / R_g   (constant, independent of slip)
#   S_C = V_C / (R_g * omega_p)
#
# With S_C ≈ 0 these reduce to pure-rolling kinematics. With S_C ≈ 1,
# omega_c ≈ omega_p and omega_b ≈ 0.

rng_seed = SEED + (0 if RUN_INDEX == "single" else int(RUN_INDEX))
rng = np.random.default_rng(rng_seed)

frames_all = np.arange(TOTAL_FRAMES)
times_all  = frames_all / FPS
dt         = 1.0 / FPS

# One synthetic video should represent one controlled slip condition.
# SC_VARIATION is a half-width: S_C=0.10 with SC_VARIATION=0.05 lives
# roughly in [0.05, 0.15], with position-dependent drift/hotspots.
smooth_win = max(1, int(FPS * 0.5))

phi_arr       = np.arange(NUM_SHELLS) * angle_step
current_angle = phi_arr[:, None] + omega_c_target * times_all[None, :]

if SC_VARIATION > 0:
    # Smooth angular slip modulation. Two broad high-slip sectors act like
    # repeatable drift/hotspot zones rather than white noise.
    hotspot_phase = rng.uniform(0, 2 * np.pi)
    hotspot_1 = np.sin(current_angle - hotspot_phase)
    hotspot_2 = np.sin(2.0 * current_angle + 0.7 * hotspot_phase)
    position_drift = 0.62 * hotspot_1 + 0.28 * hotspot_2

    # Bounded beta-distributed per-shell offset. This makes shells from the
    # same nominal condition slightly different while preserving the target.
    shell_offset = (rng.beta(2.2, 2.2, size=(NUM_SHELLS, 1)) - 0.5) * 0.70
    temporal_noise = rng.normal(0.0, 0.10, (NUM_SHELLS, TOTAL_FRAMES))

    SC_raw = SC_TARGET + SC_VARIATION * (position_drift + shell_offset + temporal_noise)
    kernel = np.ones(smooth_win) / smooth_win
    pad_left = smooth_win // 2
    pad_right = smooth_win - 1 - pad_left
    SC_arr = np.array([
        np.convolve(
            np.pad(SC_raw[i], (pad_left, pad_right), mode='edge'),
            kernel,
            mode='valid',
        )
        for i in range(NUM_SHELLS)
    ])
    SC_arr = np.clip(SC_arr, 0.0, 1.0)
else:
    SC_arr = np.full((NUM_SHELLS, TOTAL_FRAMES), SC_TARGET, dtype=float)

# beta is CONSTANT (independent of slip at C)
beta_slip_arr = np.full((NUM_SHELLS, TOTAL_FRAMES), beta)

# Circulation speed under the requested S_C condition.
omega_c_eff_arr = (omega_p / (1 + np.sin(theta_rad))) * (1 + SC_arr * np.sin(theta_rad))

omega_c_eff_mean = omega_c_eff_arr.mean(axis=0)
turntable_angle  = np.cumsum(omega_c_eff_mean) * dt

# Local shell spin under the requested S_C condition.
omega_b_eff_arr = (Rg * omega_p * (1 - SC_arr)) / (Rb * np.cos(beta_slip_arr) * (1 + np.sin(theta_rad)))

spin_angle_arr = np.cumsum(omega_b_eff_arr, axis=1) * dt

omega_c_nom = float(omega_c_eff_arr.mean())
beta_nom    = float(beta_slip_arr.mean())
omega_b_nom = float(omega_b_eff_arr.mean())

print(f"\nSlip at contact C:")
print(f"  SC_target = {SC_TARGET:.4f}  |  SC_variation_half_width = {SC_VARIATION:.4f}")
print(f"  mean omega_c_eff = {omega_c_nom:.4f} rad/s  (pure-roll {omega_c:.4f})")
print(f"  beta (constant)  = {np.degrees(beta_nom):.4f} deg")
print(f"  mean omega_b_eff = {omega_b_nom:.4f} rad/s  (pure-roll {omega_b:.4f})")
print(f"  mean SC          = {SC_arr.mean():.6f}")

# Randomizer: beta drifts ~10° per revolution
randomizer_increment_deg = 10.0
randomizer_std_deg       =  5.0
rev_counts = np.floor(turntable_angle / (2 * np.pi)).astype(int)
max_rev    = int(rev_counts[-1]) + 1

drift_per_shell = []
for i in range(NUM_SHELLS):
    increments = rng.normal(np.radians(randomizer_increment_deg),
                            np.radians(randomizer_std_deg),
                            size=max_rev)
    cum_drift = np.concatenate([[0.0], np.cumsum(increments)])
    drift_per_shell.append(cum_drift[rev_counts])

beta_per_shell = [beta_slip_arr[i] + drift_per_shell[i] for i in range(NUM_SHELLS)]

# Wobble
WOBBLE_AMP_MIN = Rb * 0.10 * WOBBLE_SCALE
WOBBLE_AMP_MAX = Rb * 0.22 * WOBBLE_SCALE
wobble_freq1   = omega_c_nom * 2.0
wobble_freq2   = omega_c_nom * 3.7

wob_amps1   = rng.uniform(WOBBLE_AMP_MIN, WOBBLE_AMP_MAX, NUM_SHELLS)
wob_phases1 = rng.uniform(0, 2 * np.pi, NUM_SHELLS)
wob_amps2   = rng.uniform(Rb * 0.03, Rb * 0.07, NUM_SHELLS)
wob_phases2 = rng.uniform(0, 2 * np.pi, NUM_SHELLS)

wobble_z_all = np.zeros((NUM_SHELLS, TOTAL_FRAMES))
for i in range(NUM_SHELLS):
    wobble_z_all[i] = (wob_amps1[i] * np.sin(wobble_freq1 * times_all + wob_phases1[i])
                     + wob_amps2[i] * np.sin(wobble_freq2 * times_all + wob_phases2[i]))

# Micro-jitter
slip_noise_all = np.zeros((NUM_SHELLS, TOTAL_FRAMES))
for i in range(NUM_SHELLS):
    slip_freq  = rng.uniform(0.5, 2.0) * omega_c_nom
    # Do not inject visible local spin into the pure-sliding limit.
    spin_jitter_scale = max(0.0, 1.0 - float(SC_arr[i].mean()))
    slip_amp   = rng.uniform(0.01, 0.04) * spin_jitter_scale
    slip_phase = rng.uniform(0, 2 * np.pi)
    slip_noise_all[i] = slip_amp * np.sin(slip_freq * times_all + slip_phase)

print(f"  max_rev = {max_rev}")

# ==========================================
# 6. ANIMATION LOOP
# ==========================================
print("Generating Animation Keyframes...")

for s in shells:
    s.rotation_mode = 'AXIS_ANGLE'

for frame in range(0, TOTAL_FRAMES):
    t = frame / FPS

    # Shell-center circulation from omega_c(S_C).
    turntable.rotation_euler[2] = float(turntable_angle[frame])
    turntable.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Per-shell spin + wobble
    for i, s in enumerate(shells):
        s.location.z = float(wobble_z_all[i, frame])
        s.keyframe_insert(data_path="location", index=2, frame=frame)

        current_beta = float(beta_per_shell[i][frame])
        spin_axis_x  = np.cos(current_beta)
        spin_axis_y  = np.sin(current_beta)
        spin_axis_z  = 0.0

        spin_angle = float(spin_angle_arr[i, frame]) + float(slip_noise_all[i, frame])

        s.rotation_axis_angle[0] = spin_angle
        s.rotation_axis_angle[1] = spin_axis_x
        s.rotation_axis_angle[2] = spin_axis_y
        s.rotation_axis_angle[3] = spin_axis_z
        s.keyframe_insert(data_path="rotation_axis_angle", frame=frame)

print("Keyframes done.")

# ==========================================
# 7. CAMERA SETUP
# ==========================================
max_radius = 0
if turntable:
    for child in turntable.children:
        dist = math.sqrt(child.location.x**2 + child.location.y**2)
        if dist > max_radius:
            max_radius = dist
    view_width = (max_radius * 2) * 1.2
else:
    view_width = 10.0

bpy.ops.object.camera_add(location=(0, -1.0, 0.5))
cam = bpy.context.active_object
cam.name = "Fixed_Camera"
scene.camera = cam

cam.rotation_euler = (math.radians(90), 0, 0)
cam.location.x = 0
cam.location.z = 0
cam.data.type = 'ORTHO'
cam.data.ortho_scale = view_width
cam.location.y = -(max_radius * 5 + 10)
cam.data.clip_start = 0.001
cam.data.clip_end = 100.0

constraint = cam.constraints.new(type='TRACK_TO')
constraint.target = turntable
constraint.track_axis = 'TRACK_NEGATIVE_Z'
constraint.up_axis = 'UP_Y'

# ==========================================
# 8. LIGHTING SETUP
# ==========================================

bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
sun_key = bpy.context.active_object
sun_key.name = "Sun_Key"
sun_key.data.energy = 0.0
sun_key.data.color = (1.0, 0.95, 0.85)
sun_key.rotation_euler = (math.radians(45), 0, 0)

bpy.ops.object.light_add(type='SUN', location=(0, 0, -5))
sun_fill = bpy.context.active_object
sun_fill.name = "Sun_Fill"
sun_fill.data.energy = 0.0
sun_fill.data.color = (0.85, 0.9, 1.0)
sun_fill.rotation_euler = (math.radians(-60), 0, 0)

bpy.ops.object.light_add(type='AREA', location=(0, Rg * 2.5, Rb * 0.375))
area_back = bpy.context.active_object
area_back.name = "Area_BackPanel"
area_back.data.shape = 'RECTANGLE'
area_back.data.size = plate_radius * 2.6
area_back.data.size_y = Rb * 3.5
area_back.data.energy = LIGHTING_ENERGY
area_back.data.color = (1.0, 0.96, 0.58)
area_back.rotation_euler = (math.radians(-90), 0, 0)

bpy.ops.object.light_add(type='AREA', location=(0, -(max(Rg * 2.5, 1.0)), Rb * 0.2))
area_fill = bpy.context.active_object
area_fill.name = "Area_Fill_Green"
area_fill.data.shape = 'RECTANGLE'
area_fill.data.size = plate_radius * 4.0
area_fill.data.size_y = Rb * 8.0
area_fill.data.energy = 6.0
area_fill.data.color = (0.22, 0.34, 0.18)
area_fill.rotation_euler = (math.radians(90), 0, 0)

# Backdrop
backdrop_y = Rg * 3.0
bpy.ops.mesh.primitive_plane_add(size=plate_radius * 12, location=(0, backdrop_y, 0))
backdrop = bpy.context.active_object
backdrop.name = "Backdrop_Glow"
backdrop.rotation_euler = (math.radians(90), 0, 0)

bd_mat = bpy.data.materials.new(name="Backdrop_Mat")
bd_mat.use_nodes = True
bd_n = bd_mat.node_tree.nodes
bd_l = bd_mat.node_tree.links
bd_n.clear()
bd_out = bd_n.new(type='ShaderNodeOutputMaterial')
bd_emit = bd_n.new(type='ShaderNodeEmission')
bd_emit.inputs['Color'].default_value = (0.02, 0.03, 0.02, 1.0)
bd_emit.inputs['Strength'].default_value = 0.0
bd_l.new(bd_emit.outputs['Emission'], bd_out.inputs['Surface'])
backdrop.data.materials.append(bd_mat)

# Gap streaks
gap_center_z = Rb * 0.375
gap_plane_y = Rg * 1.5
bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, gap_plane_y, gap_center_z))
gap_plane = bpy.context.active_object
gap_plane.name = "Gap_Streaks"
gap_plane.rotation_euler = (math.radians(90), 0, 0)
gap_plane.scale = (plate_radius * 2, Rb * 1.4, 1.0)

gap_mat = bpy.data.materials.new(name="Gap_Streaks_Mat")
gap_mat.use_nodes = True
gn = gap_mat.node_tree.nodes
gl = gap_mat.node_tree.links
gn.clear()

g_out   = gn.new(type='ShaderNodeOutputMaterial');   g_out.location   = (1850,   0)
g_coord = gn.new(type='ShaderNodeTexCoord');         g_coord.location = (-850,  120)

g_map = gn.new(type='ShaderNodeMapping');            g_map.location   = (-650,  120)
g_map.inputs['Scale'].default_value = (6.0, 0.18, 0.03)

g_n1 = gn.new(type='ShaderNodeTexNoise');            g_n1.location    = (-420,  240)
g_n1.inputs['Scale'].default_value      = 3.5
g_n1.inputs['Detail'].default_value     = 10.0
g_n1.inputs['Roughness'].default_value  = 0.9
g_n1.inputs['Distortion'].default_value = 3.5

g_n2 = gn.new(type='ShaderNodeTexNoise');            g_n2.location    = (-420,    0)
g_n2.inputs['Scale'].default_value      = 14.0
g_n2.inputs['Detail'].default_value     = 6.0
g_n2.inputs['Roughness'].default_value  = 0.75
g_n2.inputs['Distortion'].default_value = 0.9

g_mix = gn.new(type='ShaderNodeMixRGB');             g_mix.location   = (-180, 120)
g_mix.blend_type = 'ADD'
g_mix.inputs['Fac'].default_value = 0.35

g_ramp = gn.new(type='ShaderNodeValToRGB');          g_ramp.location  = (  40, 120)
g_ramp.color_ramp.interpolation = 'B_SPLINE'
g_ramp.color_ramp.elements[0].position = 0.0
g_ramp.color_ramp.elements[0].color    = (0.01, 0.03, 0.01, 1.0)
g_ramp_mid = g_ramp.color_ramp.elements.new(0.50)
g_ramp_mid.color = (0.28, 0.34, 0.10, 1.0)
g_ramp.color_ramp.elements[-1].position = 0.65
g_ramp.color_ramp.elements[-1].color    = (0.96, 0.98, 0.72, 1.0)

g_coord2 = gn.new(type='ShaderNodeTexCoord');        g_coord2.location = (-850, -130)
g_grad   = gn.new(type='ShaderNodeTexGradient');     g_grad.location   = (-650, -130)
g_grad.gradient_type = 'SPHERICAL'

g_power  = gn.new(type='ShaderNodeMath');            g_power.location  = (-450, -130)
g_power.operation = 'POWER'
g_power.use_clamp = True
g_power.inputs[1].default_value = 1.5

g_mult = gn.new(type='ShaderNodeMixRGB');            g_mult.location   = (260, 40)
g_mult.blend_type = 'MULTIPLY'
g_mult.inputs['Fac'].default_value = 1.0

g_map_slow = gn.new(type='ShaderNodeMapping');       g_map_slow.location = (-650, -330)
g_map_slow.inputs['Scale'].default_value = (1.2, 0.35, 0.1)

g_nslow = gn.new(type='ShaderNodeTexNoise');         g_nslow.location    = (-420, -330)
g_nslow.inputs['Scale'].default_value      = 1.6
g_nslow.inputs['Detail'].default_value     = 3.0
g_nslow.inputs['Roughness'].default_value  = 0.45
g_nslow.inputs['Distortion'].default_value = 0.15

g_slow_ramp = gn.new(type='ShaderNodeValToRGB');     g_slow_ramp.location = (-180, -330)
g_slow_ramp.color_ramp.interpolation = 'EASE'
g_slow_ramp.color_ramp.elements[0].position = 0.20
g_slow_ramp.color_ramp.elements[0].color    = (0.55, 0.55, 0.55, 1.0)
g_slow_ramp.color_ramp.elements[1].position = 0.85
g_slow_ramp.color_ramp.elements[1].color    = (1.10, 1.10, 1.10, 1.0)

g_map_speck = gn.new(type='ShaderNodeMapping');      g_map_speck.location = (-650, -520)
g_map_speck.inputs['Scale'].default_value = (28.0, 4.0, 1.0)

g_voro = gn.new(type='ShaderNodeTexVoronoi');        g_voro.location      = (-420, -520)
g_voro.feature = 'F1'
g_voro.distance = 'EUCLIDEAN'
g_voro.inputs['Scale'].default_value = 28.0

g_speck_ramp = gn.new(type='ShaderNodeValToRGB');    g_speck_ramp.location = (-180, -520)
g_speck_ramp.color_ramp.interpolation = 'LINEAR'
g_speck_ramp.color_ramp.elements[0].position = 0.02
g_speck_ramp.color_ramp.elements[0].color    = (0.35, 0.35, 0.35, 1.0)
g_speck_mid = g_speck_ramp.color_ramp.elements.new(0.045)
g_speck_mid.color = (0.08, 0.08, 0.08, 1.0)
g_speck_ramp.color_ramp.elements[-1].position = 0.085
g_speck_ramp.color_ramp.elements[-1].color    = (1.0, 1.0, 1.0, 1.0)

g_map_vert = gn.new(type='ShaderNodeMapping');       g_map_vert.location = (-650, -710)
g_map_vert.inputs['Scale'].default_value = (1.0, 1.6, 1.0)
g_map_vert.inputs['Location'].default_value = (0.0, 0.2, 0.0)

g_grad_lin = gn.new(type='ShaderNodeTexGradient');   g_grad_lin.location = (-420, -710)
g_grad_lin.gradient_type = 'LINEAR'

g_vert_ramp = gn.new(type='ShaderNodeValToRGB');     g_vert_ramp.location = (-180, -710)
g_vert_ramp.color_ramp.interpolation = 'B_SPLINE'
g_vert_ramp.color_ramp.elements[0].position = 0.10
g_vert_ramp.color_ramp.elements[0].color    = (0.60, 0.60, 0.60, 1.0)
g_vert_mid = g_vert_ramp.color_ramp.elements.new(0.50)
g_vert_mid.color = (1.15, 1.15, 1.15, 1.0)
g_vert_ramp.color_ramp.elements[-1].position = 0.90
g_vert_ramp.color_ramp.elements[-1].color    = (0.65, 0.65, 0.65, 1.0)

g_map_lr = gn.new(type='ShaderNodeMapping');         g_map_lr.location = (-650, -900)
g_map_lr.inputs['Scale'].default_value = (1.0, 1.0, 1.0)
g_map_lr.inputs['Location'].default_value = (0.0, 0.0, 0.0)

g_grad_lr = gn.new(type='ShaderNodeTexGradient');    g_grad_lr.location = (-420, -900)
g_grad_lr.gradient_type = 'LINEAR'

g_lr_ramp = gn.new(type='ShaderNodeValToRGB');       g_lr_ramp.location = (-180, -900)
g_lr_ramp.color_ramp.interpolation = 'B_SPLINE'
g_lr_ramp.color_ramp.elements[0].position = 0.18
g_lr_ramp.color_ramp.elements[0].color = (0.28, 0.28, 0.28, 1.0)
g_lr_ramp.color_ramp.elements[1].position = 0.92
g_lr_ramp.color_ramp.elements[1].color    = (1.0, 1.0, 1.0, 1.0)

g_map_fine = gn.new(type='ShaderNodeMapping');       g_map_fine.location = (-650, -1090)
g_map_fine.inputs['Scale'].default_value = (55.0, 9.0, 1.0)

g_nfine = gn.new(type='ShaderNodeTexNoise');         g_nfine.location = (-420, -1090)
g_nfine.inputs['Scale'].default_value = 22.0
g_nfine.inputs['Detail'].default_value = 12.0
g_nfine.inputs['Roughness'].default_value = 0.65
g_nfine.inputs['Distortion'].default_value = 0.1

g_fine_ramp = gn.new(type='ShaderNodeValToRGB');     g_fine_ramp.location = (-180, -1090)
g_fine_ramp.color_ramp.interpolation = 'LINEAR'
g_fine_ramp.color_ramp.elements[0].position = 0.20
g_fine_ramp.color_ramp.elements[0].color    = (0.88, 0.88, 0.88, 1.0)
g_fine_ramp.color_ramp.elements[1].position = 0.85
g_fine_ramp.color_ramp.elements[1].color    = (1.03, 1.03, 1.03, 1.0)

g_mult_vert = gn.new(type='ShaderNodeMixRGB');       g_mult_vert.location = (500, -80)
g_mult_vert.blend_type = 'MULTIPLY'
g_mult_vert.inputs['Fac'].default_value = 1.0

g_mult_slow = gn.new(type='ShaderNodeMixRGB');       g_mult_slow.location = (740, -80)
g_mult_slow.blend_type = 'MULTIPLY'
g_mult_slow.inputs['Fac'].default_value = 1.0

g_mult_speck = gn.new(type='ShaderNodeMixRGB');      g_mult_speck.location = (980, -80)
g_mult_speck.blend_type = 'MULTIPLY'
g_mult_speck.inputs['Fac'].default_value = 1.0

g_mult_lr = gn.new(type='ShaderNodeMixRGB');         g_mult_lr.location = (1220, -80)
g_mult_lr.blend_type = 'MULTIPLY'
g_mult_lr.inputs['Fac'].default_value = 1.0

g_mult_fine = gn.new(type='ShaderNodeMixRGB');       g_mult_fine.location = (1460, -80)
g_mult_fine.blend_type = 'MULTIPLY'
g_mult_fine.inputs['Fac'].default_value = 1.0

g_emit = gn.new(type='ShaderNodeEmission');          g_emit.location = (1660, 120)

g_strength_mult  = gn.new(type='ShaderNodeMath');    g_strength_mult.location  = (980, 120)
g_strength_mult.operation = 'MULTIPLY'
g_strength_mult.use_clamp = False
g_strength_mult.inputs[0].default_value = 15

g_strength_mult2 = gn.new(type='ShaderNodeMath');    g_strength_mult2.location = (1220, 120)
g_strength_mult2.operation = 'MULTIPLY'
g_strength_mult2.use_clamp = False

g_strength_mult3 = gn.new(type='ShaderNodeMath');    g_strength_mult3.location = (1460, 120)
g_strength_mult3.operation = 'MULTIPLY'
g_strength_mult3.use_clamp = False

# Links
gl.new(g_coord.outputs['Object'],  g_map.inputs['Vector'])
gl.new(g_map.outputs['Vector'],    g_n1.inputs['Vector'])
gl.new(g_map.outputs['Vector'],    g_n2.inputs['Vector'])
gl.new(g_n1.outputs['Fac'],        g_mix.inputs['Color1'])
gl.new(g_n2.outputs['Fac'],        g_mix.inputs['Color2'])
gl.new(g_mix.outputs['Color'],     g_ramp.inputs['Fac'])
gl.new(g_ramp.outputs['Color'],    g_mult.inputs['Color1'])

gl.new(g_coord2.outputs['Object'], g_grad.inputs['Vector'])
gl.new(g_grad.outputs['Fac'],      g_power.inputs[0])
gl.new(g_power.outputs['Value'],   g_mult.inputs['Color2'])

gl.new(g_coord.outputs['Object'],  g_map_slow.inputs['Vector'])
gl.new(g_map_slow.outputs['Vector'], g_nslow.inputs['Vector'])
gl.new(g_nslow.outputs['Fac'],     g_slow_ramp.inputs['Fac'])

gl.new(g_coord.outputs['Object'],  g_map_speck.inputs['Vector'])
gl.new(g_map_speck.outputs['Vector'], g_voro.inputs['Vector'])
gl.new(g_voro.outputs['Distance'], g_speck_ramp.inputs['Fac'])

gl.new(g_coord.outputs['Object'],  g_map_vert.inputs['Vector'])
gl.new(g_map_vert.outputs['Vector'], g_grad_lin.inputs['Vector'])
gl.new(g_grad_lin.outputs['Fac'],  g_vert_ramp.inputs['Fac'])

gl.new(g_coord.outputs['Object'],  g_map_lr.inputs['Vector'])
gl.new(g_map_lr.outputs['Vector'], g_grad_lr.inputs['Vector'])
gl.new(g_grad_lr.outputs['Fac'],   g_lr_ramp.inputs['Fac'])

gl.new(g_coord.outputs['Object'],  g_map_fine.inputs['Vector'])
gl.new(g_map_fine.outputs['Vector'], g_nfine.inputs['Vector'])
gl.new(g_nfine.outputs['Fac'],     g_fine_ramp.inputs['Fac'])

gl.new(g_mult.outputs['Color'],       g_mult_vert.inputs['Color1'])
gl.new(g_vert_ramp.outputs['Color'],  g_mult_vert.inputs['Color2'])

gl.new(g_mult_vert.outputs['Color'],  g_mult_slow.inputs['Color1'])
gl.new(g_slow_ramp.outputs['Color'],  g_mult_slow.inputs['Color2'])

gl.new(g_mult_slow.outputs['Color'],  g_mult_speck.inputs['Color1'])
gl.new(g_speck_ramp.outputs['Color'], g_mult_speck.inputs['Color2'])

gl.new(g_mult_speck.outputs['Color'], g_mult_lr.inputs['Color1'])
gl.new(g_lr_ramp.outputs['Color'],    g_mult_lr.inputs['Color2'])

gl.new(g_mult_lr.outputs['Color'],    g_mult_fine.inputs['Color1'])
gl.new(g_fine_ramp.outputs['Color'],  g_mult_fine.inputs['Color2'])

gl.new(g_mult_fine.outputs['Color'], g_emit.inputs['Color'])

gl.new(g_slow_ramp.outputs['Color'],   g_strength_mult.inputs[1])
gl.new(g_speck_ramp.outputs['Color'],  g_strength_mult2.inputs[0])
gl.new(g_strength_mult.outputs['Value'], g_strength_mult2.inputs[1])
gl.new(g_strength_mult2.outputs['Value'], g_strength_mult3.inputs[0])
gl.new(g_lr_ramp.outputs['Color'],       g_strength_mult3.inputs[1])

gl.new(g_strength_mult3.outputs['Value'], g_emit.inputs['Strength'])

gl.new(g_emit.outputs['Emission'], g_out.inputs['Surface'])
gap_plane.data.materials.append(gap_mat)

# Bottom glow plane
bottom_glow_z = -Rb * 3.8
bpy.ops.mesh.primitive_plane_add(size=plate_radius * 10, location=(0, Rg * 2.0, bottom_glow_z))
bot_glow = bpy.context.active_object
bot_glow.name = "Bottom_Glow"
bot_glow.rotation_euler = (math.radians(90), 0, 0)

botg_mat = bpy.data.materials.new(name="Bottom_Glow_Mat")
botg_mat.use_nodes = True
botg_n = botg_mat.node_tree.nodes
botg_l = botg_mat.node_tree.links
botg_n.clear()

botg_out = botg_n.new(type='ShaderNodeOutputMaterial')
botg_out.location = (980, 0)
botg_emit = botg_n.new(type='ShaderNodeEmission')
botg_emit.location = (780, 0)
botg_emit.inputs['Strength'].default_value = 0.18

botg_mult_col = botg_n.new(type='ShaderNodeMixRGB')
botg_mult_col.location = (580, 40)
botg_mult_col.blend_type = 'MULTIPLY'
botg_mult_col.inputs['Fac'].default_value = 1.0
botg_mult_col.inputs['Color1'].default_value = (0.05, 0.07, 0.025, 1.0)

botg_coord = botg_n.new(type='ShaderNodeTexCoord')
botg_coord.location = (-700, 0)
botg_map = botg_n.new(type='ShaderNodeMapping')
botg_map.location = (-500, 0)
botg_map.inputs['Scale'].default_value = (1.0, 1.0, 1.0)

botg_sep = botg_n.new(type='ShaderNodeSeparateXYZ')
botg_sep.location = (-300, 0)

botg_ramp = botg_n.new(type='ShaderNodeValToRGB')
botg_ramp.location = (-80, 0)
botg_ramp.color_ramp.interpolation = 'B_SPLINE'
botg_ramp.color_ramp.elements[0].position = 0.35
botg_ramp.color_ramp.elements[0].color = (0.01, 0.01, 0.01, 1.0)
botg_ramp.color_ramp.elements[1].position = 0.92
botg_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

botg_rgb2bw = botg_n.new(type='ShaderNodeRGBToBW')
botg_rgb2bw.location = (140, -80)

botg_strength_mult = botg_n.new(type='ShaderNodeMath')
botg_strength_mult.location = (360, -80)
botg_strength_mult.operation = 'MULTIPLY'
botg_strength_mult.inputs[0].default_value = 0.18

botg_l.new(botg_coord.outputs['Generated'], botg_map.inputs['Vector'])
botg_l.new(botg_map.outputs['Vector'], botg_sep.inputs['Vector'])
botg_l.new(botg_sep.outputs['X'], botg_ramp.inputs['Fac'])
botg_l.new(botg_ramp.outputs['Color'], botg_mult_col.inputs['Color2'])
botg_l.new(botg_mult_col.outputs['Color'], botg_emit.inputs['Color'])
botg_l.new(botg_ramp.outputs['Color'], botg_rgb2bw.inputs['Color'])
botg_l.new(botg_rgb2bw.outputs['Val'], botg_strength_mult.inputs[1])
botg_l.new(botg_strength_mult.outputs['Value'], botg_emit.inputs['Strength'])
botg_l.new(botg_emit.outputs['Emission'], botg_out.inputs['Surface'])

bot_glow.data.materials.append(botg_mat)

# ==========================================
# 9. WORLD & COMPOSITOR
# ==========================================

if scene.world and scene.world.node_tree:
    bg_nodes = scene.world.node_tree.nodes
    bg = bg_nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.02, 0.01, 0.0, 1)
        bg.inputs[1].default_value = 0.05

try:
    scene.use_nodes = True
    tree = scene.node_tree
    if tree:
        tree.nodes.clear()
        rl = tree.nodes.new('CompositorNodeRLayers')
        rl.location = (0, 0)
        glare = tree.nodes.new('CompositorNodeGlare')
        glare.location = (300, 0)
        glare.glare_type = 'FOG_GLOW'
        glare.threshold = 0.45
        glare.size = 7
        comp = tree.nodes.new('CompositorNodeComposite')
        comp.location = (600, 0)
        tree.links.new(rl.outputs[0], glare.inputs[0])
        tree.links.new(glare.outputs[0], comp.inputs[0])
except Exception:
    pass

# ==========================================
# 10. DATASET EXPORT
# ==========================================
print("Generating dataset...")

angle_step_shells = 2 * np.pi / NUM_SHELLS
world_pos = np.zeros((NUM_SHELLS, TOTAL_FRAMES, 3))
for i in range(NUM_SHELLS):
    local_angle  = i * angle_step_shells
    world_angle  = local_angle + turntable_angle
    world_pos[i, :, 0] = Rg * np.cos(world_angle)
    world_pos[i, :, 1] = Rg * np.sin(world_angle)
    world_pos[i, :, 2] = wobble_z_all[i]

world_vel  = np.gradient(world_pos, dt, axis=1)
tang_speed = Rg * omega_c_eff_arr
speed_mag  = np.linalg.norm(world_vel, axis=2)
circ_rpm   = omega_c_eff_arr * 60.0 / (2 * np.pi)
spin_rpm   = omega_b_eff_arr * 60.0 / (2 * np.pi)

view_height = view_width * RENDER_H / RENDER_W
img_cx = (world_pos[:, :, 0] / (view_width  / 2) + 1.0) / 2.0 * RENDER_W
img_cy = (1.0 - (world_pos[:, :, 2] / (view_height / 2) + 1.0) / 2.0) * RENDER_H
img_r  = Rb / view_width * RENDER_W

bb_x1 = img_cx - img_r
bb_y1 = img_cy - img_r
bb_x2 = img_cx + img_r
bb_y2 = img_cy + img_r

in_frame = ((img_cx > 0) & (img_cx < RENDER_W) &
            (img_cy > 0) & (img_cy < RENDER_H))

# CSV
csv_path = os.path.join(OUTPUT_DIR, f"{RUN_ID}_gt.csv")
with open(csv_path, 'w', newline='') as fcsv:
    w = csv.writer(fcsv)
    w.writerow([
        'frame', 'time_s', 'shell_id',
        'world_x_m', 'world_y_m', 'world_z_m',
        'vel_x_mps', 'vel_y_mps', 'vel_z_mps',
        'speed_mps', 'tang_speed_mps',
        'circ_rpm', 'spin_rpm', 'head_rpm',
        'omega_c_radps', 'omega_b_radps', 'spin_beta_rad',
        'SC',
        'img_cx_px', 'img_cy_px', 'img_r_px',
        'bb_x1', 'bb_y1', 'bb_x2', 'bb_y2', 'in_frame',
    ])
    for i in range(NUM_SHELLS):
        for f in range(TOTAL_FRAMES):
            w.writerow([
                f,
                round(float(times_all[f]),        6),
                i,
                round(float(world_pos[i, f, 0]),  6),
                round(float(world_pos[i, f, 1]),  6),
                round(float(world_pos[i, f, 2]),  6),
                round(float(world_vel[i, f, 0]),  6),
                round(float(world_vel[i, f, 1]),  6),
                round(float(world_vel[i, f, 2]),  6),
                round(float(speed_mag[i, f]),     6),
                round(float(tang_speed[i, f]),    6),
                round(float(circ_rpm[i, f]),      4),
                round(float(spin_rpm[i, f]),      4),
                OMEGA_P_RPM,
                round(float(omega_c_eff_arr[i, f]), 6),
                round(float(omega_b_eff_arr[i, f]), 6),
                round(float(beta_per_shell[i][f]),  6),
                round(float(SC_arr[i, f]),  6),
                round(float(img_cx[i, f]),  2),
                round(float(img_cy[i, f]),  2),
                round(img_r,                2),
                round(float(bb_x1[i, f]),   2),
                round(float(bb_y1[i, f]),   2),
                round(float(bb_x2[i, f]),   2),
                round(float(bb_y2[i, f]),   2),
                int(bool(in_frame[i, f])),
            ])

# NPZ
npz_path = os.path.join(OUTPUT_DIR, f"{RUN_ID}_gt.npz")
np.savez_compressed(
    npz_path,
    frames    = frames_all,
    times_s   = times_all,
    world_pos = world_pos,
    world_vel = world_vel,
    speed_mps      = speed_mag,
    tang_speed_mps = tang_speed,
    omega_c_radps = omega_c_eff_arr,
    omega_b_radps = omega_b_eff_arr,
    spin_beta_rad = np.array(beta_per_shell),
    circ_rpm = circ_rpm,
    spin_rpm = spin_rpm,
    head_rpm = np.array([OMEGA_P_RPM]),
    SC  = SC_arr,
    img_cx   = img_cx,
    img_cy   = img_cy,
    img_r_px = np.array([img_r]),
    bb_x1    = bb_x1,
    bb_y1    = bb_y1,
    bb_x2    = bb_x2,
    bb_y2    = bb_y2,
    in_frame = in_frame,
    render_wh  = np.array([RENDER_W, RENDER_H]),
    view_wh_m  = np.array([view_width, view_height]),
    Rg_m       = np.array([Rg]),
    Rb_m       = np.array([Rb]),
    theta_deg  = np.array([theta]),
)

n_rows = NUM_SHELLS * TOTAL_FRAMES

# Metadata JSON
metadata = {
    "run_id": RUN_ID,
    "description": "No-slip-at-A/B C-slip sweep: V-groove drive with idle top pad",
    "parameters": {
        "regime": "controlled_c_slip",
        "slip_type": "S_C",
        "contact_constraints": "V_A=V_B=0; V_C=S_C*R_g*omega_p",
        "omega_p_rpm": OMEGA_P_RPM,
        "num_shells": NUM_SHELLS,
        "motion_blur_shutter": MOTION_BLUR_SHUTTER,
        "lighting_energy": LIGHTING_ENERGY,
        "sc_target": SC_TARGET,
        "sc_variation": SC_VARIATION,
        "sc_mean_base": SC_MEAN_BASE,
        "run_group": RUN_GROUP,
        "wobble_scale": WOBBLE_SCALE,
        "duration_sec": DURATION_SEC,
        "total_frames": TOTAL_FRAMES,
        "fps": FPS,
        "seed": rng_seed,
    },
    "geometry": {
        "Rb_m": Rb,
        "Rg_m": Rg,
        "theta_deg": theta,
        "render_w": RENDER_W,
        "render_h": RENDER_H,
        "view_width_m": view_width,
        "view_height_m": float(view_height),
    },
    "derived_kinematics": {
        "omega_c_pure_radps": float(omega_c),
        "omega_b_pure_radps": float(omega_b),
        "beta_pure_deg": float(np.degrees(beta)),
        "omega_c_target_radps": float(omega_c_target),
        "omega_b_target_radps": float(omega_b_target),
        "omega_c_eff_mean_radps": float(omega_c_nom),
        "omega_b_eff_mean_radps": float(omega_b_nom),
        "mean_SC": float(SC_arr.mean()),
        "min_SC": float(SC_arr.min()),
        "max_SC": float(SC_arr.max()),
        "std_SC": float(SC_arr.std()),
        "Ri_m": float(Ri),
        "Ro_m": float(Ro),
    },
    "dataset_stats": {
        "total_frames": TOTAL_FRAMES,
        "total_rows": n_rows,
        "img_r_px": float(img_r),
    },
    "output_files": {
        "gt_csv": os.path.basename(csv_path),
        "gt_npz": os.path.basename(npz_path),
        "frames_dir": "frames/",
    }
}

meta_path = os.path.join(OUTPUT_DIR, f"{RUN_ID}_metadata.json")
with open(meta_path, 'w') as _mf:
    json.dump(metadata, _mf, indent=2)

# Render output path
scene.render.filepath = os.path.join(OUTPUT_DIR, "frames", f"{RUN_ID}_")
try:
    scene.render.image_settings.file_format = 'PNG'
except TypeError:
    try:
        scene.render.image_settings.file_format = 'AVI_RAW'
        scene.render.image_settings.file_format = 'PNG'
    except TypeError:
        print("WARNING: Cannot set PNG programmatically.")
scene.render.image_settings.color_mode = 'RGB'
scene.render.image_settings.compression = 15

print(f"\n{'='*50}")
print(f"C-SLIP {OMEGA_P_RPM} RPM, S_C={SC_TARGET:.3f} — SETUP COMPLETE!")
print(f"{'='*50}")
print(f"  Shells: {NUM_SHELLS}  |  Frames: {TOTAL_FRAMES}  |  Duration: {DURATION_SEC}s")
print(f"  SC target/mean: {SC_TARGET:.4f} / {SC_arr.mean():.6f}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  CSV : {csv_path}  ({n_rows} rows)")
print(f"  NPZ : {npz_path}")
print(f"  META: {meta_path}")
print(f"  Render: {scene.render.filepath}")
if AUTO_RENDER:
    print("\n  >> AUTO_RENDER=1, rendering animation now")
else:
    print(f"\n  >> Press Ctrl+F12 to render this single setup")
    print(f"  >> Default script run launches the full 20-video batch")
print(f"{'='*50}")

if AUTO_RENDER:
    bpy.ops.render.render(animation=True)
    print(f"Rendered animation frames to: {os.path.join(OUTPUT_DIR, 'frames')}")
