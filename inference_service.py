# inference_service.py

import io
import json
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

import uvicorn
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# Load YOLO once
device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
yolo = YOLO("yolov8n.pt").to(device)

def run_auto(pil_img, conf):
    arr = np.array(pil_img)[:, :, ::-1]
    dets = []
    for pred in yolo.predict(arr, conf=conf):
        for b in pred.boxes:
            x1,y1,x2,y2 = b.xyxy[0].tolist()
            dets.append({"xyxy":[x1,y1,x2,y2],"conf":float(b.conf),"label":pred.names[int(b.cls[0])]})
    return dets

def annotate_frame(pil, yolo_conf, mapping, manual):
    ann = pil.copy(); draw = ImageDraw.Draw(ann)
    try: font = ImageFont.truetype("arial.ttf",24)
    except:  font = ImageFont.load_default()

    # Auto (red)
    for it in run_auto(pil, yolo_conf):
        if it["conf"]<yolo_conf: continue
        lbl = mapping.get(it["label"], it["label"])
        x1,y1,x2,y2 = it["xyxy"]
        txt = f"{lbl} {it['conf']:.2f}"
        draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
        bb = draw.textbbox((x1,y1), txt, font=font)
        draw.rectangle(bb, fill="red")
        draw.text((x1,bb[1]), txt, fill="white", font=font)

    # Manual (blue)
    for mb in manual:
        try:
            x1,y1,x2,y2 = map(float, mb["xyxy"])
            if any(np.isnan([x1,y1,x2,y2])): continue
            x1,x2 = sorted((x1,x2)); y1,y2 = sorted((y1,y2))
            draw.rectangle([x1,y1,x2,y2], outline="blue", width=3)
            lbl = mb.get("label","manual")
            bb = draw.textbbox((x1,y1), lbl, font=font)
            draw.rectangle(bb, fill="blue")
            draw.text((x1,bb[1]), lbl, fill="white", font=font)
        except:
            continue

    return ann
@app.post("/get_initial_labels")
async def get_initial_labels(
    file: UploadFile = File(...),
    yolo_conf: float = Form(0.5),
    label_map: str = Form(""),
    seconds: int = Form(3)  # retained signature but not used
):
    """
    Extracts unique labels detected from the entire video.
    """
    data = await file.read()
    mapping = json.loads(label_map or "{}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(data)
    tmp.close()

    cap = cv2.VideoCapture(tmp.name)
    labels_found = set()

    # Process every frame until end-of-video
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets = run_auto(img, yolo_conf)
        for d in dets:
            lbl = mapping.get(d["label"], d["label"])
            labels_found.add(lbl)

    cap.release()
    return JSONResponse({"labels": sorted(labels_found)})


@app.post("/predict_image")
async def predict_image(file:UploadFile=File(...),
                        yolo_conf:float=Form(0.5),
                        label_map:str=Form("")):
    data = await file.read()
    try: img = Image.open(io.BytesIO(data)).convert("RGB")
    except: raise HTTPException(400,"Invalid image")
    mapping = json.loads(label_map or "{}")
    dets = run_auto(img, yolo_conf)
    for it in dets: it["label"] = mapping.get(it["label"], it["label"])
    return JSONResponse({"yolov8": dets})

@app.post("/predict_image_manual")
async def predict_image_manual(
    file:UploadFile=File(...),
    yolo_conf:float=Form(0.5),
    label_map:str=Form(""),
    manual_boxes:str=Form(""),
):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    mapping = json.loads(label_map or "{}")
    mbs = json.loads(manual_boxes or "[]")
    ann = annotate_frame(img, yolo_conf, mapping, mbs)
    buf = io.BytesIO(); ann.save(buf,"PNG"); buf.seek(0)
    return StreamingResponse(io.BytesIO(buf.getvalue()), media_type="image/png")

@app.post("/predict_video")
async def predict_video(file:UploadFile=File(...),
                        yolo_conf:float=Form(0.5),
                        label_map:str=Form(""),):
    data = await file.read()
    mapping = json.loads(label_map or "{}")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(data); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok: break
        frames.append(frm)
    cap.release()
    if not frames: raise HTTPException(400,"No frames")
    h,w = frames[0].shape[:2]
    fps = cv2.VideoCapture(tmp.name).get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4"); vw = cv2.VideoWriter(outf.name, fourcc, fps, (w,h))
    for frm in frames:
        pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        ann = annotate_frame(pil, yolo_conf, mapping, [])
        vw.write(cv2.cvtColor(np.array(ann), cv2.COLOR_RGB2BGR))
    vw.release()
    return StreamingResponse(open(outf.name,"rb"), media_type="video/mp4")

def numpy_iou(box, boxes):
    box   = np.array(box, dtype=float)
    boxes = np.array(boxes, dtype=float)
    x1 = np.maximum(box[0], boxes[:,0]); y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2]); y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, x2-x1)*np.maximum(0, y2-y1)
    area1 = (box[2]-box[0])*(box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    return inter/(area1+area2-inter+1e-6)

@app.post("/predict_video_manual")
async def predict_video_manual(file: UploadFile = File(...),
    yolo_conf:float=Form(0.5), label_map:str=Form(""), manual_boxes:str=Form(""),):
    data = await file.read()
    mapping = json.loads(label_map or "{}")
    seeds   = json.loads(manual_boxes or "[]")
    if not seeds: raise HTTPException(400,"No manual boxes provided")

    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".mp4")
    tmp.write(data); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frames=[]
    while True:
        ok,frm=cap.read();
        if not ok: break
        frames.append(frm)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not frames: raise HTTPException(400,"No frames")

    num_frames = len(frames)
    manual_pf = [[] for _ in range(num_frames)]

    by_label = {}
    for mb in seeds:
        lbl = mb.get("label","manual")
        by_label.setdefault(lbl,[]).append(mb)

    def is_present(idx, lbl, box):
        img = Image.fromarray(cv2.cvtColor(frames[idx],cv2.COLOR_BGR2RGB))
        for d in run_auto(img, yolo_conf):
            dl = mapping.get(d["label"], d["label"])
            if dl==lbl and numpy_iou(box,[d["xyxy"]])[0]>0.3:
                return True
        return False

    for lbl, box_list in by_label.items():
        box_list = sorted(box_list, key=lambda x:int(x["frame"]))
        if len(box_list)==1:
            s=box_list[0]; f0=int(s["frame"])
            if not(0<=f0<num_frames): continue
            x1,y1,x2,y2=map(float,s["xyxy"])
            manual_pf[f0].append({"xyxy":[x1,y1,x2,y2],"label":lbl})
            # backward
            tr=cv2.TrackerCSRT_create()
            tr.init(frames[f0],(int(x1),int(y1),int(x2-x1),int(y2-y1)))
            for f in range(f0-1,-1,-1):
                ok,bb=tr.update(frames[f])
                if not ok: break
                b=[float(bb[0]),float(bb[1]),float(bb[0]+bb[2]),float(bb[1]+bb[3])]
                if not is_present(f,lbl,b): break
                manual_pf[f].append({"xyxy":b,"label":lbl})
            # forward
            tr=cv2.TrackerCSRT_create()
            tr.init(frames[f0],(int(x1),int(y1),int(x2-x1),int(y2-y1)))
            for f in range(f0+1,num_frames):
                ok,bb=tr.update(frames[f])
                if not ok: break
                b=[float(bb[0]),float(bb[1]),float(bb[0]+bb[2]),float(bb[1]+bb[3])]
                if not is_present(f,lbl,b): break
                manual_pf[f].append({"xyxy":b,"label":lbl})
            continue
        # multi seed
        first=box_list[0]; f0=int(first["frame"])
        if not(0<=f0<num_frames): continue
        x1,y1,x2,y2=map(float,first["xyxy"])
        tr=cv2.TrackerCSRT_create()
        tr.init(frames[f0],(int(x1),int(y1),int(x2-x1),int(y2-y1)))
        manual_pf[f0].append({"xyxy":[x1,y1,x2,y2],"label":lbl})
        for ns in box_list[1:]:
            fn=int(ns["frame"])
            if fn>=num_frames: break
            for f in range(f0+1,fn+1):
                ok,bb=tr.update(frames[f])
                if not ok: break
                b=[float(bb[0]),float(bb[1]),float(bb[0]+bb[2]),float(bb[1]+bb[3])]
                if not is_present(f,lbl,b): break
                manual_pf[f].append({"xyxy":b,"label":lbl})
            x1,y1,x2,y2=map(float,ns["xyxy"])
            tr=cv2.TrackerCSRT_create()
            tr.init(frames[fn],(int(x1),int(y1),int(x2-x1),int(y2-y1)))
            manual_pf[fn].append({"xyxy":[x1,y1,x2,y2],"label":lbl})
            f0=fn

    # render video
    h,w=frames[0].shape[:2]
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    outf=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); outf.close()
    vw=cv2.VideoWriter(outf.name,fourcc,fps,(w,h))
    for i,frm in enumerate(frames):
        pil=Image.fromarray(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
        ann=annotate_frame(pil,yolo_conf,mapping,manual_pf[i])
        vw.write(cv2.cvtColor(np.array(ann),cv2.COLOR_RGB2BGR))
    vw.release()

    return StreamingResponse(open(outf.name,"rb"),media_type="video/mp4")

if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)
