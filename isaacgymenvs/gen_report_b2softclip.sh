#!/bin/bash
# 生成 cgsmooth_B2_softclip 单+多 报告(同之前格式: cmp 参考|策略 / policy / ref_physics_vs_policy)。
source ~/miniconda3/etc/profile.d/conda.sh; conda activate dextrack
cd /home/liangh/DexTrack/isaacgymenvs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liangh/miniconda3/envs/dextrack/lib
GPU=0
RV=/home/liangh/DexTrack/report_videos
SM=$RV/cgsmooth_B2_softclip_single; MM=$RV/cgsmooth_B2_softclip_multi; mkdir -p $SM $MM
HE="HAND_EMA_COEF=0.4 WUJI_DATA_DIR=./data/GRAB_Tracking_PK_WUJI_FPOS_v1/data numEnvs=100"
# ckpt
cp "$(ls -t logs/grab_single_wuji_pinall3_cgsmooth_B2_softclip/ori_grab_s2_cubesmall_inspect_1/*/ckpt/best_*.pth|head -1)" /tmp/ck_b2sc_single.pth
cp "$(ls -t logs/grab_multiple_wuji_pinall3_cg_smooth_grabct2_softclip/wuji_cubesmall_pinall3/*/ckpt/best_*.pth|head -1)" /tmp/ck_b2sc_multi.pth
echo "[rep] ckpts copied"

run_test(){  # $1=script $2=seq -> echoes rollout path
  touch /tmp/rep_marker
  env $HE bash scripts/$1 $GPU $2 /tmp/$3 True > /tmp/rep_test_$2.log 2>&1
  find logs_test -name "ts_to_hand_obj_obs_reset_1.npy" -newer /tmp/rep_marker 2>/dev/null | head -1
}
render(){ # $1=src $2=seq $3=out  (策略面板, cam_follow hand cam_smooth21, 选高举env)
  env=$(python3 -c "import numpy as np;d=np.load('$1',allow_pickle=True).item();ts=sorted(k for k in d if isinstance(k,int));op=np.array([d[t]['object_pose'][:,:3] for t in ts]);gp=np.array([d[t]['goal_pose_ref_np'][:,:3] for t in ts]);tr=np.linalg.norm(op-gp,axis=-1).mean(0);print(int(np.argmin(tr)))" 2>/dev/null)
  [ -z "$env" ] && env=1
  python wuji_isaacgym_playback.py --src "$1" --env $env --hand wuji --obj_code "$2" --gpu $GPU \
     --cam_scale 0.5 --cam_follow hand --cam_smooth 21 --out "$3" >/dev/null 2>&1
  echo $env
}
succ(){ python3 -c "import numpy as np;d=np.load('$1',allow_pickle=True).item();ts=sorted(k for k in d if isinstance(k,int));op=np.array([d[t]['object_pose'][:,:3] for t in ts]);gp=np.array([d[t]['goal_pose_ref_np'][:,:3] for t in ts]);e=int(np.argmin(np.linalg.norm(op-gp,axis=-1).mean(0)));tr=np.linalg.norm(op[:,e]-gp[:,e],axis=-1).mean();lift=op[:,e,2].max()-op[0,e,2];print('success' if (tr<0.05 and lift>0.05) else 'fail')" 2>/dev/null || echo fail; }

stitch(){ # $1=ref_mp4 $2=pol_mp4 $3=out_base $4=pol_title
python3 - "$1" "$2" "$3" "$4" << 'PY'
import sys,imageio.v2 as im,numpy as np
from PIL import Image,ImageDraw,ImageFont
ref,pol,out,ptit=sys.argv[1:5]
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24)
except:font=ImageFont.load_default()
def L(m):return [f[...,:3] for f in im.get_reader(m)]
def T(a,t):
 img=Image.fromarray(a.astype('uint8'));d=ImageDraw.Draw(img);w=d.textlength(t,font=font)
 d.text(((img.width-w)//2,12),t,fill=(20,20,20),font=font);return np.array(img)
A,B=L(ref),L(pol);n=min(len(A),len(B))
o=np.array([np.concatenate([T(A[i],'Reference (kinematic)'),T(B[i],ptit)],1) for i in range(n)],dtype='uint8')
im.mimsave(out+'.mp4',o,fps=20,quality=8);im.mimsave(out+'.gif',o[::2],duration=0.1)
print('stitched',out)
PY
}

echo "[rep] === MULTI: 6 seq ==="
for seq in ori_grab_s2_cubesmall_inspect_1 ori_grab_s2_cubesmall_pass_1 ori_grab_s6_cubesmall_pass_1 ori_grab_s8_cubesmall_inspect_1 ori_grab_s10_cubesmall_pass_1 ori_grab_s1_cubesmall_offhand_1; do
  short=$(echo $seq|sed -E 's#ori_grab_(s[0-9]+)_cubesmall_([a-z]+)_?[0-9]*#\1_\2#')
  echo "[rep] multi $short ..."
  R=$(run_test run_tracking_headless_grab_multiple_wuji_test.sh $seq ck_b2sc_multi.pth)
  [ -z "$R" ] && { echo "  !! $short 测试无rollout"; continue; }
  oc=$(succ "$R")
  refk=$(ls -t logs_test/refk_${seq}/*/*/ts_to_hand_obj_obs_reset_1.npy|head -1)
  e=$(render "$R" "$seq" /tmp/pol_$short.mp4)
  python wuji_isaacgym_playback.py --src "$refk" --env $e --hand wuji --obj_code "$seq" --gpu $GPU --cam_scale 0.5 --cam_follow hand --cam_smooth 21 --out /tmp/ref_$short.mp4 >/dev/null 2>&1
  stitch /tmp/ref_$short.mp4 /tmp/pol_$short.mp4 $MM/cmp_${short}_${oc} 'Policy: cgsmooth_B2_softclip'
done

echo "[rep] === SINGLE ==="
R=$(run_test run_tracking_headless_grab_single_wuji_test.sh ori_grab_s2_cubesmall_inspect_1 ck_b2sc_single.pth)
if [ -n "$R" ]; then
  e=$(render "$R" ori_grab_s2_cubesmall_inspect_1 $SM/cgsmooth_B2_softclip_policy.mp4)
  python3 -c "import imageio.v2 as im;v=[f[...,:3] for f in im.get_reader('$SM/cgsmooth_B2_softclip_policy.mp4')];im.mimsave('$SM/cgsmooth_B2_softclip_policy.gif',v[::2],duration=0.1)"
  refk=$(ls -t logs_test/refk_ori_grab_s2_cubesmall_inspect_1/*/*/ts_to_hand_obj_obs_reset_1.npy|head -1)
  python wuji_isaacgym_playback.py --src "$refk" --env $e --hand wuji --obj_code ori_grab_s2_cubesmall_inspect_1 --gpu $GPU --cam_scale 0.5 --cam_follow hand --cam_smooth 21 --out /tmp/sref.mp4 >/dev/null 2>&1
  stitch /tmp/sref.mp4 $SM/cgsmooth_B2_softclip_policy.mp4 $SM/ref_physics_vs_policy 'Policy: cgsmooth_B2_softclip'
fi
echo "[rep] DONE" | tee -a /tmp/gen_report_b2sc.log
ls $SM $MM
