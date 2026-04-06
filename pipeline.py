"""
pipeline.py — Railway-safe: nothing runs at import time
"""

import os, sys, time, json, hashlib, gc, textwrap
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from dotenv import load_dotenv
load_dotenv()

MODEL_MIOU          = 0.0
_segmentor_instance = None
_graph_instance     = None


def _get_segmentor():
    global _segmentor_instance, MODEL_MIOU
    if _segmentor_instance is None:
        from segmentor import Segmentor
        _segmentor_instance = Segmentor()
        MODEL_MIOU = _segmentor_instance.miou
    return _segmentor_instance


def _get_graph():
    global _graph_instance
    if _graph_instance is None:
        try:
            from terrain_graph import TerrainGraph
            _graph_instance = TerrainGraph()
            print("✅ TigerGraph connected")
        except Exception as e:
            print(f"⚠️  TigerGraph unavailable: {e}")
            _graph_instance = _FallbackGraph()
    return _graph_instance


class _FallbackGraph:
    def upload(self, image_id, mask):
        return {"patches":0,"edges":0,"risks":0}
    def find_path(self, image_id):
        return {"hops":15,"cost":0.0,
                "waypoints":[[8,15-i] for i in range(16)]}
    def query_knowledge(self, image_id):
        return {"similar_terrains":[]}


CLASS_NAMES = ["Dense_Vegetation","Dry_Vegetation","Ground_Objects",
               "Rocks","Landscape","Sky"]
TRAV = {"Dense_Vegetation":0.45,"Dry_Vegetation":0.75,"Ground_Objects":0.30,
        "Rocks":0.15,"Landscape":0.95,"Sky":0.00}
RISK = {"Dense_Vegetation":"MED","Dry_Vegetation":"LOW",
        "Ground_Objects":"MED_HIGH","Rocks":"HIGH","Landscape":"LOW","Sky":"N/A"}
PALETTE = np.array([[34,139,34],[210,180,140],[139,90,43],
                    [128,128,128],[205,133,63],[135,206,235]],dtype=np.uint8)
OUTPUT_ROOT = Path("pipeline_outputs")
OUTPUT_ROOT.mkdir(exist_ok=True)


def _colorize(mask):
    out = np.zeros((*mask.shape,3),dtype=np.uint8)
    for i,rgb in enumerate(PALETTE): out[mask==i]=rgb
    return out

def _overlay(img,mask,alpha=0.45):
    color = _colorize(mask); h,w = mask.shape
    orig  = cv2.resize(img,(w,h)) if img.shape[:2]!=(h,w) else img.copy()
    return cv2.addWeighted(orig.astype(np.uint8),1-alpha,
                           color.astype(np.uint8),alpha,0)

def _save(arr,path): Image.fromarray(arr.astype(np.uint8)).save(path)

def _class_dist(mask):
    total = mask.size
    return {n:float(np.sum(mask==i))/total
            for i,n in enumerate(CLASS_NAMES) if np.sum(mask==i)>0}

def _scene_trav(dist):
    s = sum(TRAV.get(k,.5)*v for k,v in dist.items())
    d = sum(dist.values())
    return round(s/d,4) if d>0 else 0.5

def _build_risk_zones(mask):
    h,w=mask.shape; G=16; ph,pw=h//G,w//G; zones=[]
    for ci,name in enumerate(CLASS_NAMES):
        sev=RISK.get(name,"LOW")
        if sev in ("LOW","N/A"): continue
        count=rows=cols=0
        for gr in range(G):
            for gc_ in range(G):
                patch=mask[gr*ph:(gr+1)*ph,gc_*pw:(gc_+1)*pw]
                dom=int(np.bincount(patch.flatten(),minlength=NUM_CLASSES).argmax())
                if dom==ci: count+=1;rows+=gr;cols+=gc_
        if count>0:
            zones.append({"attributes":{
                "class_name":name,"severity":sev,
                "patch_count":count,
                "center_row":round(rows/count),
                "center_col":round(cols/count)}})
    order={"HIGH":0,"MED_HIGH":1,"MED":2}
    return sorted(zones,key=lambda z:order.get(z["attributes"]["severity"],9))

NUM_CLASSES = len(CLASS_NAMES)

def _save_path_viz(img,mask,path_info,save_path):
    blend=_overlay(img,mask); h,w=blend.shape[:2]; G=16
    pw_g,ph_g=w//G,h//G; wps=path_info.get("waypoints",[])
    if len(wps)>=2:
        pts=np.array(wps,dtype=np.int32)
        for i in range(len(pts)-1):
            p1=(int(pts[i][0]*pw_g+pw_g//2),int(pts[i][1]*ph_g+ph_g//2))
            p2=(int(pts[i+1][0]*pw_g+pw_g//2),int(pts[i+1][1]*ph_g+ph_g//2))
            cv2.line(blend,p1,p2,(0,255,120),3,cv2.LINE_AA)
        cv2.circle(blend,(int(pts[0][0]*pw_g+pw_g//2),
                          int(pts[0][1]*ph_g+ph_g//2)),8,(0,255,120),-1)
        cv2.circle(blend,(int(pts[-1][0]*pw_g+pw_g//2),
                          int(pts[-1][1]*ph_g+ph_g//2)),8,(255,80,0),-1)
    _save(blend,save_path)


def run_pipeline(image_path, use_tta=False, save_outputs=True):
    t0       = time.time()
    img_path = Path(image_path)
    image_id = "img_"+hashlib.md5(img_path.read_bytes()).hexdigest()[:8]
    out_dir  = OUTPUT_ROOT/image_id
    out_dir.mkdir(parents=True,exist_ok=True)

    print(f"\n{'='*60}\nProcessing: {img_path.name}  →  {image_id}\n{'='*60}")
    img_rgb = np.array(Image.open(img_path).convert("RGB"))

    # 1 — Segmentation
    t1=time.time(); print("[1/5] Segmenting...")
    seg=_get_segmentor(); mask=seg.predict(img_rgb)
    dist=_class_dist(mask); trav=_scene_trav(dist)
    t_seg=round(time.time()-t1,2); print(f"      {t_seg}s")
    for n,f in sorted(dist.items(),key=lambda x:-x[1]):
        print(f"      {n:<25}: {f*100:5.1f}%")

    if save_outputs:
        _save(img_rgb,             out_dir/"original.png")
        _save(_colorize(mask),     out_dir/"segmented.png")
        _save(_overlay(img_rgb,mask),out_dir/"overlay.png")
    gc.collect()

    # 2 — Graph
    t1=time.time(); print("[2/5] Building terrain graph...")
    graph=_get_graph(); ginfo=graph.upload(image_id,mask)
    t_graph=round(time.time()-t1,2)
    print(f"      {t_graph}s  patches={ginfo.get('patches',0)}  edges={ginfo.get('edges',0)}")

    # 3 — Path
    t1=time.time(); print("[3/5] Dijkstra path...")
    pinfo=graph.find_path(image_id)
    hops=pinfo.get("hops",pinfo.get("hop_count",0))
    cost=pinfo.get("cost",pinfo.get("total_cost",0.0))
    path_out={"hop_count":int(hops),"total_cost":round(float(cost),4),
              "waypoints":pinfo.get("waypoints",[])}
    t_path=round(time.time()-t1,2); print(f"      {t_path}s  hops={hops}")
    if save_outputs: _save_path_viz(img_rgb,mask,pinfo,out_dir/"path.png")

    # 4 — Knowledge
    t1=time.time(); print("[4/5] Knowledge query...")
    know=graph.query_knowledge(image_id)
    risk_zones=_build_risk_zones(mask)
    raw_sim=know.get("similar_terrains",[])
    similar_out=[{"attributes":{
        "avg_traversability":t.get("avg_traversability",t.get("traversability",0)),
        "dominant_class":t.get("dominant_class","?"),
        "miou_achieved":t.get("miou_achieved",0),
        "image_id":t.get("image_id","?")}}
        if isinstance(t,dict) and "attributes" not in t else t
        for t in raw_sim]
    t_know=round(time.time()-t1,2)
    print(f"      {t_know}s  risks={len(risk_zones)}  similar={len(similar_out)}")

    # 5 — LLM
    t1=time.time(); print("[5/5] AI briefing...")
    try:
        from explainer import generate_navigation_briefing, generate_system_explanation
        briefing=generate_navigation_briefing(
            class_dist=dist, path_result=path_out,
            risk_zones=risk_zones, similar_terrains=similar_out[:3],
            avg_traversability=trav)
        explanation=generate_system_explanation(
            class_dist=dist, path_result=path_out,
            risk_zones=risk_zones, model_miou=MODEL_MIOU)
    except Exception as e:
        print(f"  ⚠️  LLM: {e}")
        dom=max(dist,key=dist.get) if dist else "terrain"
        briefing=(f"Terrain dominated by {dom} ({dist.get(dom,0)*100:.0f}%). "
                  f"Traversability: {trav*100:.0f}%. "
                  f"Safest path: {hops} hops. "
                  f"{len(risk_zones)} risk zone(s) detected.")
        explanation=briefing
    t_llm=round(time.time()-t1,2); print(f"      {t_llm}s")

    total=round(time.time()-t0,2)
    result={
        "image_id":image_id,"class_dist":dist,"path":path_out,
        "risk_zones":risk_zones,"similar":similar_out,
        "avg_trav":trav,"briefing":briefing,"explanation":explanation,
        "total_time":total,"model_miou":getattr(seg,"miou",MODEL_MIOU),
        "graph_patches":ginfo.get("patches",0),
        "graph_edges":ginfo.get("edges",0),
        "timing":{"segmentation":t_seg,"terrain_graph":t_graph,
                  "dijkstra":t_path,"knowledge":t_know,"llm":t_llm},
    }
    if save_outputs:
        with open(out_dir/"result.json","w") as f:
            json.dump(result,f,indent=2,default=str)
    print(f"\n✅ Done in {total}s\n📋 {textwrap.fill(briefing,68)}\n")
    return result


if __name__=="__main__":
    if len(sys.argv)<2: print("Usage: python pipeline.py <image>"); sys.exit(1)
    run_pipeline(sys.argv[1])