"""Final pass: re-gate every generated TOPO sequence (corrected relchange metric),
sort kept/rejected, merge FPOS penetration, print per-object summary + tables.
Generation is metric-independent so this only re-reads artifacts (no re-retarget).
"""
import argparse, json, os, glob, subprocess, shutil
import numpy as np

ROOT = "/data/home/liangheng/DexTrack"
RETPY = f"{ROOT}/wuji_retarget_pen_min/.pixi/envs/default/bin/python"
TOPO = f"{ROOT}/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_TOPO_v1"
FPOSD = f"{ROOT}/isaacgymenvs/data/GRAB_Tracking_PK_WUJI_FPOS_v1"


def obj_of(tag): return tag[len("ori_grab_"):].split("_", 1)[1].split("_")[0]


def main():
    tags = sorted(os.path.basename(x)[len("wuji_passive_active_info_"):-len("_nf_300.npy")]
                  for x in glob.glob(f"{FPOSD}/data/*.npy") if "_cubesmall_" in x or "_cup_" in x or "_apple_" in x)
    # un-reject: move everything back to data/ so we gate from a clean state
    for sub, suf in [("data", "_nf_300.npy"), ("contact_grab2", "_contact.npy")]:
        for f in glob.glob(f"{TOPO}/rejected/{sub}/*"):
            shutil.move(f, f"{TOPO}/{sub}/{os.path.basename(f)}")
    os.makedirs(f"{TOPO}/rejected/data", exist_ok=True); os.makedirs(f"{TOPO}/rejected/contact_grab2", exist_ok=True)

    # FPOS penetration (one pass, reuse across tags)
    fpos_pen = {}
    fp_out = "/tmp/topo_gen/fpos_pen.jsonl"
    if not os.path.exists(fp_out):
        tf = "/tmp/topo_gen/all_tags.txt"; open(tf, "w").write("\n".join(tags))
        subprocess.run([RETPY, f"{ROOT}/wuji_pipeline/fpos_pen_batch.py", "--tags", tf, "--out", fp_out],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for line in open(fp_out):
        r = json.loads(line); fpos_pen[r["tag"]] = r["fpos_pen_max_mm"]

    rows = []
    for tag in tags:
        topo_npy = f"{TOPO}/data/wuji_passive_active_info_{tag}_nf_300.npy"
        motion = f"/tmp/topo_gen/{tag}/export/motions/{tag}/motion.npz"
        ctopo = f"{TOPO}/contact_grab2/{tag}_contact.npy"
        cfpos = f"{FPOSD}/contact_grab2/{tag}_contact.npy"
        sim = f"/tmp/topo_gen/{tag}/sim.npy"
        if not (os.path.exists(topo_npy) and os.path.exists(motion)):
            rows.append({"tag": tag, "PASS": False, "reason": "not_generated"}); continue
        out = subprocess.run([RETPY, f"{ROOT}/wuji_pipeline/gate_check.py",
                              "--topo", topo_npy, "--fpos", f"{FPOSD}/data/wuji_passive_active_info_{tag}_nf_300.npy",
                              "--motion", motion, "--contact-topo", ctopo, "--contact-fpos", cfpos,
                              "--sim", sim, "--tag", tag], capture_output=True, text=True)
        try:
            r = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception:
            rows.append({"tag": tag, "PASS": False, "reason": "gate_error", "err": out.stderr[-200:]}); continue
        r["fpos_pen_max_mm"] = fpos_pen.get(tag)
        rows.append(r)
        if not r["PASS"]:
            for src, sub in [(topo_npy, "data"), (ctopo, "contact_grab2")]:
                if os.path.exists(src): shutil.move(src, f"{TOPO}/rejected/{sub}/{os.path.basename(src)}")

    json.dump(rows, open("/tmp/topo_gen/final_rows.json", "w"), indent=1)

    # ---- report ----
    def why(r):
        if r.get("reason"): return r["reason"]
        rs = []
        if r["orient_diff_median_deg"] >= 20: rs.append(f"orient_med={r['orient_diff_median_deg']:.1f}")
        if r["pen_max_mm"] > 3: rs.append(f"pen={r['pen_max_mm']:.1f}")
        if r["joint_viol_frames"]: rs.append("jointlim")
        if r["nan"]: rs.append("nan")
        if not r["b2_flag_equal_fpos"]: rs.append("b2flag")
        return ",".join(rs) or "?"

    print("\n================ PER-SEQUENCE ================")
    print(f"{'tag':38s} {'PASS':4} {'pen':>5} {'fpos':>5} {'omed':>5} {'relc':>5} {'b2':>3}  reason")
    for r in rows:
        if r.get("reason") in ("not_generated", "gate_error"):
            print(f"{r['tag']:38s} FAIL    -     -     -     -    -   {why(r)}"); continue
        print(f"{r['tag']:38s} {'Y' if r['PASS'] else 'N':4} {r['pen_max_mm']:5.2f} "
              f"{(r['fpos_pen_max_mm'] or -1):5.1f} {r['orient_diff_median_deg']:5.1f} {r['relchange_corr']:5.2f} "
              f"{'Y' if r['b2_flag_equal_fpos'] else 'N':3}  {'' if r['PASS'] else why(r)}")

    print("\n================ PER-OBJECT SUMMARY ================")
    for obj in ["cubesmall", "cup", "apple"]:
        og = [r for r in rows if obj_of(r["tag"]) == obj]
        okp = [r for r in og if r.get("PASS")]
        pen_t = [r["pen_max_mm"] for r in okp]; pen_f = [r["fpos_pen_max_mm"] for r in okp if r["fpos_pen_max_mm"]]
        print(f"{obj:10s}: {len(okp)}/{len(og)} pass | TOPO pen mean={np.mean(pen_t):.2f}mm max={np.max(pen_t):.2f}mm | "
              f"FPOS pen(same seqs) mean={np.mean(pen_f):.1f}mm max={np.max(pen_f):.1f}mm")
    allok = [r for r in rows if r.get("PASS")]
    allpt = [r["pen_max_mm"] for r in allok]; allpf = [r["fpos_pen_max_mm"] for r in allok if r["fpos_pen_max_mm"]]
    print(f"{'TOTAL':10s}: {len(allok)}/{len(rows)} pass | TOPO pen mean={np.mean(allpt):.2f}mm | FPOS pen mean={np.mean(allpf):.1f}mm")
    print(f"kept in data/: {len(glob.glob(f'{TOPO}/data/*.npy'))}  contact: {len(glob.glob(f'{TOPO}/contact_grab2/*.npy'))}  "
          f"rejected: {len(glob.glob(f'{TOPO}/rejected/data/*.npy'))}")


if __name__ == "__main__":
    main()
