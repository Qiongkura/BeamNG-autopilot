import sys, time
sys.path.insert(0, ".")
import numpy as np
from beamng_autopilot.connector import BeamNGConnector
from beamng_autopilot import config
from beamng_autopilot.roadnet import RoadNetwork

conn = BeamNGConnector("italy", "etk800",
                       port=config.runtime_port("tech"),
                       home=config.runtime_home("tech"))
conn.open(launch=True)
try:
    conn.attach_vehicle(already_open=True, vid="thePlayer")
except Exception:
    conn.load_scenario()
    conn.step(60)
    conn.attach_vehicle(already_open=True, vid="ego")
rn = RoadNetwork()
t0 = time.time()
while not rn.ready and time.time() - t0 < 90:
    if rn.build(conn.bng):
        break
    time.sleep(1.0)
for i in range(5):
    st = conn.get_state()
    xyz = rn.nearest_node_xyz(st.pos[:2]) if rn.ready else None
    print("pos_z=%.2f  road_z=%s  diff=%.2f"
          % (float(st.pos[2]),
             ("%.2f" % xyz[2]) if xyz else "n/a",
             float(st.pos[2]) - xyz[2] if xyz else float("nan")), flush=True)
    time.sleep(0.3)
conn.close()