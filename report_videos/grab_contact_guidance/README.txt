GRAB 真值接触 vs wuji投影接触 (contact guidance 接触点来源对比)

视频 (ref 参考回放, 接触点标在真实物体上, 每指一色: 拇红/食绿/中蓝/无名橙/小紫):
  *_contact_AvsB.mp4  -- ● 实心圆 = B (GRAB原数据真值接触顶点质心)
                          ◇ 空心菱形 = A (wuji重定向指尖投影到物体表面)
  cubesmall_contact_Btruth_only.mp4 -- 只叠 B 真值, 更干净

图 (object-local 三视图: 全物体 / 放大 / 单帧 片→质心):
  *_AvsB_mesh.png

结论: B 修正两类 A 的启发式误差 -- 点位(cubesmall拇指1.9cm) + flag时序(flute过检30-40%).
帧对齐完美(normMSE=0). 是ref不是policy. 细节见 docs/grab_contact_guidance_plan.md
