import sys, numpy as np, math
sys.path.insert(0, '.')
from beamng_autopilot.connector import BeamNGConnector
from beamngpy.sensors import Camera
from beamng_autopilot_tech.providers import CAMERA_POS, CAMERA_DIR, CAMERA_UP, CAMERA_FOV_DEG
from beamng_autopilot.vision.segmentation import Segmenter
from beamng_autopilot.vision.lanes import MarkingSmoother
from beamng_autopilot.lane import pair_lane_markings
from beamng_autopilot.vision.projection import default_camera

conn = BeamNGConnector(port=64257)
conn.open(launch=False)
conn.attach_vehicle(already_open=True)
cam = Camera('diag_lane', conn.bng, conn.vehicle, requested_update_time=0.05,
             pos=CAMERA_POS, dir=CAMERA_DIR, up=CAMERA_UP,
             resolution=(1076, 806), field_of_view_y=CAMERA_FOV_DEG,
             near_far_planes=(0.05, 150.0), is_using_shared_memory=True,
             is_render_colours=True, is_render_annotations=False,
             is_render_instance=False, is_render_depth=False, is_visualised=False)
seg = Segmenter('logs/m5_seg/seg_model/best.pt')
sm = MarkingSmoother()
cm = default_camera(1076, 806)

for i in range(12):
    conn.step(10)
    with conn.io_lock:
        data = cam.poll()
    img = np.ascontiguousarray(np.asarray(data['colour'], dtype=np.uint8))
    st = conn.get_state()
    gz = float(st.pos[2]); heading = float(st.heading)
    raw = seg.detect_lines(img, cm, st.pos, heading, ground_z=gz)
    marks = sm.update(raw, cm, st.pos, heading, ground_z=gz,
                      warmup=(i < 4), speed=float(st.speed))
    dbg = {}
    frame = pair_lane_markings(marks, st.pos, heading, fwd=st.dir, debug=dbg)
    # 各标线的横向位置
    fwd2 = np.asarray(st.dir[:2], dtype=float); fn = np.linalg.norm(fwd2)
    if fn > 1e-9: fwd2 = fwd2 / fn
    left = np.array([-fwd2[1], fwd2[0]])
    pos2 = np.asarray(st.pos[:2], dtype=float)
    lats = []
    for mk in marks:
        w = np.asarray(mk.world, dtype=float)[:, :2]
        rel = w - pos2
        lats.append(float(np.median(rel @ left)))
    fl = fr = fc = None
    if frame is not None:
        def mlat(poly):
            if poly is None or len(poly) < 2: return None
            w = np.asarray(poly, dtype=float)[:, :2]
            return float(np.median((w - pos2) @ left))
        fl = mlat(frame.left); fr = mlat(frame.right); fc = mlat(frame.center)
    print(f'f{i:02d} heading={heading:.2f} raw={len(raw)} marks={len(marks)} '
          f'mark_lats={[round(v,2) for v in lats[:6]]}')
    if frame is not None:
        print(f'     frame: left={fl} right={fr} center={fc} '
              f'paired={frame.paired} conf={frame.confidence:.2f} '
              f'src={frame.sources} width={frame.width:.1f} mode={dbg.get("mode")}')
    else:
        print(f'     frame: None  mode={dbg.get("mode")}')
with conn.io_lock:
    cam.remove()
conn.close()