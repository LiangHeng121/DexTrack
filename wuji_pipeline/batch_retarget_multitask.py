"""Batch-retarget the multi-task sequences (cubesmall+flute) to wuji references
(no-offset). Orchestrates the 4-step pipeline across the two conda envs via
`conda run`, per sequence, with the subject's real hand vtemp (from GRAB
subject_meshes). Output -> default WUJI_v1 dir (no-offset). Skips existing.

Run from repo root in ANY env (uses conda run internally):
    python wuji_pipeline/batch_retarget_multitask.py [--list assets/inst_tag_list_obj_cubesmall_flute.npy]
"""
import argparse, os, re, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--list", default="assets/inst_tag_list_obj_cubesmall_flute.npy")
ap.add_argument("--out-dir", default="isaacgymenvs/data/GRAB_Tracking_PK_WUJI_v1/data")
ap.add_argument("--scale", default="1.25")
args = ap.parse_args()

OUTDIR = args.out_dir
CONFIG = "wuji/wuji-retargeting/example/config/retarget_manus_right.yaml"
ALLEGRO_DIR = "isaacgymenvs/data/GRAB_Tracking_PK_reduced_300/data"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs("wuji_pipeline/out", exist_ok=True)


def run(env, *cmd, capture=False):
    full = ["conda", "run", "-n", env, "python", *cmd]
    r = subprocess.run(full, capture_output=capture, text=True)
    if r.returncode != 0:
        out = (r.stdout or "") + (r.stderr or "")
        raise RuntimeError(f"[{env}] {' '.join(cmd[:2])} failed:\n{out[-1500:]}")
    return r.stdout if capture else ""


def retarget_one(tag):
    # tag = ori_grab_s1_cubesmall_inspect_1_nf_300
    body = tag[len("ori_grab_"):]                 # s1_cubesmall_inspect_1_nf_300
    subj, seq = body.split("_", 1)                # s1 , cubesmall_inspect_1_nf_300
    seq = seq[:-len("_nf_300")]                   # cubesmall_inspect_1
    grab_npz = f"GRAB/unzipped/grab/{subj}/{seq}.npz"
    allegro = f"{ALLEGRO_DIR}/passive_active_info_{tag}.npy"
    out = f"{OUTDIR}/wuji_passive_active_info_{tag}.npy"
    kp = f"wuji_pipeline/out/{subj}_{seq}_mano_kp21.npy"
    qpos = f"wuji_pipeline/out/{subj}_{seq}_wuji_qpos20.npy"
    if not os.path.exists(grab_npz):
        return "skip(no grab npz)"
    if not os.path.exists(allegro):
        return "skip(no allegro ref)"
    if os.path.exists(out):
        return "skip(exists)"
    # vtemp from npz rhand.vtemp
    vt = np.load(grab_npz, allow_pickle=True)["rhand"].item()["vtemp"]
    vtemp = f"GRAB/unzipped/{vt}"

    # 0 keypoints (dextrack)
    run("dextrack", "wuji_pipeline/grab_to_mano_keypoints.py",
        "--grab-npz", grab_npz, "--vtemp", vtemp, "--out", kp)
    # 1 retarget scale 1.25 (wuji-retarget)
    run("wuji-retarget", "wuji_pipeline/retarget_keypoints_to_wuji.py",
        "--kp", kp, "--config", CONFIG, "--out", qpos, "--scale", args.scale)
    # 2 assemble (dextrack) -> capture crop
    so = run("dextrack", "wuji_pipeline/assemble_wuji_reference.py",
             "--grab-npz", grab_npz, "--allegro-ref", allegro,
             "--wuji-qpos", qpos, "--out", out, capture=True)
    mcrop = re.search(r"best crop GRAB\[(\d+):(\d+)\]", so)
    c0, c1 = mcrop.group(1), mcrop.group(2)
    # 3 rederive global (wuji-retarget)
    run("wuji-retarget", "wuji_pipeline/rederive_wuji_global.py",
        "--grab", grab_npz, "--allegro", allegro, "--kp21", kp,
        "--out", out, "--c0", c0, "--c1", c1)
    # numpy 2.x -> 1.24 roundtrip so dextrack (training/render) can load it
    tmp = f"/tmp/wuji_rt_{tag}.npz"
    run("wuji-retarget", "-c",
        f"import numpy as np;d=np.load('{out}',allow_pickle=True).item();"
        f"np.savez('{tmp}',ot=d['object_transl'],oq=d['object_rot_quat'],rs=d['robot_delta_states_weights_np'])")
    run("dextrack", "-c",
        f"import numpy as np;z=np.load('{tmp}');"
        f"np.save('{out}',dict(object_transl=z['ot'],object_rot_quat=z['oq'],robot_delta_states_weights_np=z['rs']))")
    os.path.exists(tmp) and os.remove(tmp)
    return f"OK (crop {c0}:{c1})"


tags = sorted(np.load(args.list, allow_pickle=True).item().keys())
print(f"[batch] {len(tags)} sequences -> {OUTDIR}", flush=True)
ok = skip = fail = 0
for i, tag in enumerate(tags, 1):
    try:
        r = retarget_one(tag)
        if r.startswith("OK"): ok += 1
        else: skip += 1
        print(f"[{i}/{len(tags)}] {tag}: {r}", flush=True)
    except Exception as e:
        fail += 1
        print(f"[{i}/{len(tags)}] {tag}: FAIL {str(e)[:300]}", flush=True)
print(f"[batch done] ok={ok} skip={skip} fail={fail}", flush=True)
