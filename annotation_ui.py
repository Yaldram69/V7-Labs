# annotation_ui.py

import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io, json, tempfile, cv2, os
from streamlit_drawable_canvas import st_canvas

API_BASE         = "http://localhost:8000"
IMAGE_AUTO_URL   = f"{API_BASE}/predict_image"
IMAGE_MANUAL_URL = f"{API_BASE}/predict_image_manual"
VIDEO_AUTO_URL   = f"{API_BASE}/predict_video"
VIDEO_MANUAL_URL = f"{API_BASE}/predict_video_manual"

SCALE_FACTOR     = 2
MAX_CANVAS_WIDTH = 600
BOX_STROKE_IMG   = 3
BOX_STROKE_VID   = 4

st.set_page_config(page_title="Annotation UI", layout="wide")
st.title("Annotation UI")
st.write("Upload an image or video to auto‑annotate, then rename labels or refine manually.")

if "label_map" not in st.session_state:
    st.session_state.label_map = {}

if "auto_vid" not in st.session_state:
    st.session_state.auto_vid = None

if "final_vid" not in st.session_state:
    st.session_state.final_vid = None

def draw_boxes(pil_img, dets, color, width):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()
    for it in dets:
        x1, y1, x2, y2 = it["xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        txt = it["label"] + (f" {it.get('conf', 0):.2f}" if it.get("conf") is not None else "")
        bb = draw.textbbox((x1, y1), txt, font=font)
        draw.rectangle(bb, fill=color)
        draw.text((x1, bb[1]), txt, fill="white", font=font)

mode   = st.radio("Mode:", ["Image", "Video"])
upfile = st.file_uploader("Choose file", type=["jpg","jpeg","png","mp4","avi","mov"])
if not upfile:
    st.stop()

data = upfile.getvalue()
ext  = upfile.name.rsplit(".", 1)[-1].lower()
mime = {"mp4":"video/mp4","avi":"video/avi","mov":"video/quicktime"}.get(ext, f"image/{ext}")

# ── IMAGE FLOW ────────────────────────────────────────────────────────────────
if mode == "Image" and ext in ["jpg","jpeg","png"]:
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    w, h = pil.size
    disp_w     = min(w * SCALE_FACTOR, MAX_CANVAS_WIDTH)
    total_scale= disp_w / w
    disp_h     = int(h * total_scale)
    disp_img   = pil.resize((disp_w, disp_h), Image.LANCZOS)

    # Auto‑detect
    with st.spinner("Auto‑detecting…"):
        resp = requests.post(
            IMAGE_AUTO_URL,
            files={"file": (upfile.name, data, mime)},
            data={"yolo_conf":0.5, "label_map": json.dumps(st.session_state.label_map)},
        )
        resp.raise_for_status()
        raw = resp.json().get("yolov8", [])

    # Map labels
    mapped = [{
        "xyxy": it["xyxy"],
        "conf": it["conf"],
        "label": st.session_state.label_map.get(it["label"], it["label"])
    } for it in raw]

    # Draw
    scaled = [{
        "xyxy": [c * total_scale for c in it["xyxy"]],
        "conf": it["conf"],
        "label": it["label"]
    } for it in mapped]
    auto_img = disp_img.copy()
    draw_boxes(auto_img, scaled, "red", BOX_STROKE_IMG)
    st.image(auto_img, caption="Auto‑annotated image", width=disp_w)

    # Download auto‑annotated
    buf = io.BytesIO()
    auto_img.save(buf, format="PNG")
    st.download_button("Download Auto‑Annotated Image", data=buf.getvalue(), file_name="auto.png", mime="image/png")

    # Rename inputs
    st.write("🔤 Rename auto‑labels:")
    for lbl in sorted({it["label"] for it in mapped}):
        new = st.text_input(f"Rename '{lbl}' →", lbl, key=f"map_{lbl}")
        if new and new != lbl:
            st.session_state.label_map[lbl] = new

    # Manual draw
    st.write("✍️ Draw manual boxes:")
    canvas = st_canvas(
        fill_color="rgba(0,255,0,0.3)", stroke_width=BOX_STROKE_IMG, stroke_color="blue",
        background_image=auto_img, width=disp_w, height=disp_h, drawing_mode="rect", key="img_canvas",
    )

    manual = []
    if canvas.json_data and canvas.json_data.get("objects"):
        for i, obj in enumerate(canvas.json_data["objects"]):
            lbl = st.text_input(f"Label {i+1}:", "", key=f"lbl_img_{i}")
            x1 = obj["left"]      / total_scale
            y1 = obj["top"]       / total_scale
            x2 = (obj["left"]+obj["width"])  / total_scale
            y2 = (obj["top"] +obj["height"]) / total_scale
            manual.append({"xyxy":[x1,y1,x2,y2], "label":lbl})

    if st.button("Apply Manual to Image"):
        with st.spinner("Re‑drawing…"):
            resp2 = requests.post(
                IMAGE_MANUAL_URL,
                files={"file":(upfile.name, data, mime)},
                data={
                    "yolo_conf":0.5,
                    "label_map": json.dumps(st.session_state.label_map),
                    "manual_boxes": json.dumps(manual)
                },
            )
            resp2.raise_for_status()
            final = Image.open(io.BytesIO(resp2.content))
        final_disp = final.resize((disp_w, disp_h), Image.LANCZOS)
        st.image(final_disp, caption="Manual‑refined image", width=disp_w)

        buf2 = io.BytesIO()
        final.save(buf2, format="PNG")
        st.download_button("Download Refined Image", data=buf2.getvalue(), file_name="refined.png", mime="image/png")

# ── VIDEO FLOW ────────────────────────────────────────────────────────────────
elif mode == "Video" and ext in ["mp4","avi","mov"]:
    st.video(data)

    # Allow renaming labels (by looking at first frame detections)
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    cap = cv2.VideoCapture(io.BytesIO(data) if False else tempfile.NamedTemporaryFile(delete=False,suffix=".mp4").name)
    # Actually reuse the upload bytes:
    tmpv = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmpv.write(data); tmpv.close()
    cap = cv2.VideoCapture(tmpv.name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frm = cap.read()
    cap.release()
    if not ok:
        st.error("Could not read first frame"); st.stop()

    pil0 = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    buf0 = io.BytesIO(); pil0.save(buf0, "PNG"); buf0.seek(0)
    with st.spinner("Detecting labels…"):
        resp0 = requests.post(
            IMAGE_AUTO_URL,
            files={"file":("frame.png", buf0.getvalue(), "image/png")},
            data={"yolo_conf":0.5, "label_map": json.dumps(st.session_state.label_map)},
        )
        resp0.raise_for_status()
        raw0 = resp0.json().get("yolov8", [])

    mapped0 = [{
        "xyxy": it["xyxy"],
        "conf": it["conf"],
        "label": st.session_state.label_map.get(it["label"], it["label"])
    } for it in raw0]

    # Show first frame with auto labels
    disp_w = min(pil0.width * SCALE_FACTOR, MAX_CANVAS_WIDTH)
    disp_h = int(pil0.height * (disp_w / pil0.width))
    img0 = pil0.resize((disp_w, disp_h), Image.LANCZOS)
    draw_boxes(img0, [
        {**it, "xyxy":[c*(disp_w/pil0.width) for c in it["xyxy"]]} for it in mapped0
    ], "red", BOX_STROKE_VID)
    st.image(img0, caption="Auto‑detected on first frame", width=disp_w)

    st.write("🔤 Rename auto‑labels:")
    for lbl in sorted({it["label"] for it in mapped0}):
        new = st.text_input(f"Rename '{lbl}' →", lbl, key=f"vid_map_{lbl}")
        if new and new != lbl:
            st.session_state.label_map[lbl] = new

    # Auto‑annotate entire video
    if st.button("Auto‑Annotate Video"):
        with st.spinner("Processing…"):
            resp = requests.post(
                VIDEO_AUTO_URL,
                files={"file": (upfile.name, data, mime)},
                data={"yolo_conf":0.5, "label_map": json.dumps(st.session_state.label_map)},
            )
            resp.raise_for_status()
            st.session_state.auto_vid = resp.content
        st.video(st.session_state.auto_vid)
        st.download_button(
            "Download Auto‑Annotated Video",
            data=st.session_state.auto_vid,
            file_name="auto_annotated.mp4",
            mime="video/mp4"
        )

        # Extract frames for slider preview
        tmp_auto = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_auto.write(st.session_state.auto_vid)
        tmp_auto.close()
        cap = cv2.VideoCapture(tmp_auto.name)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.session_state.auto_total = total
        cap.release()

    # Slider to preview any auto‑annotated frame
    if st.session_state.auto_vid is not None:
        idx = st.slider("Preview auto‑annotated frame", 0, st.session_state.auto_total-1, 0)
        tmp_auto = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_auto.write(st.session_state.auto_vid)
        tmp_auto.close()
        cap = cv2.VideoCapture(tmp_auto.name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frm = cap.read()
        cap.release()
        if ok:
            pilf = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
            st.image(pilf, caption=f"Frame {idx}", width=disp_w)

    st.write("---")
    st.write("✍️ Manual‑refine video by drawing on key frames:")

    if "video_manual_boxes" not in st.session_state:
        st.session_state.video_manual_boxes = []

    total = int(cv2.VideoCapture(tmpv.name).get(cv2.CAP_PROP_FRAME_COUNT))
    f0    = st.slider("Frame to annotate", 0, total-1, 0, key="manual_frame")
    cap   = cv2.VideoCapture(tmpv.name)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, frm = cap.read()
    cap.release()
    if not ok:
        st.error("Could not extract frame"); st.stop()

    pil = Image.fromarray(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    disp_w = min(pil.width * SCALE_FACTOR, MAX_CANVAS_WIDTH)
    disp_h = int(pil.height * (disp_w / pil.width))
    disp_img = pil.resize((disp_w, disp_h), Image.LANCZOS)
    st.image(disp_img, caption=f"Draw on frame {f0}", width=disp_w)

    canvas = st_canvas(
        fill_color="rgba(0,255,0,0.3)", stroke_width=BOX_STROKE_VID, stroke_color="blue",
        background_image=disp_img, width=disp_w, height=disp_h,
        drawing_mode="rect", key="vid_canvas",
    )

    manual_current = []
    if canvas.json_data and canvas.json_data.get("objects"):
        for i, obj in enumerate(canvas.json_data["objects"]):
            lbl = st.text_input(f"Label {i+1}:", "", key=f"vid_lbl_{i}")
            x1 = obj["left"] / (disp_w / pil.width)
            y1 = obj["top"]  / (disp_h / pil.height)
            x2 = (obj["left"]+obj["width"])  / (disp_w / pil.width)
            y2 = (obj["top"] +obj["height"]) / (disp_h / pil.height)
            manual_current.append({"frame": f0, "xyxy":[x1,y1,x2,y2], "label":lbl})

    if st.button("Add Annotation"):
        if manual_current:
            st.session_state.video_manual_boxes.extend(manual_current)
            st.success(f"Added {len(manual_current)} seed(s). Total: {len(st.session_state.video_manual_boxes)}")
        else:
            st.warning("Draw at least one box")

    if st.session_state.video_manual_boxes:
        st.write("**Current manual seeds:**")
        st.json(st.session_state.video_manual_boxes)
        if st.button("Clear All Seeds"):
            st.session_state.video_manual_boxes = []
            st.experimental_rerun()

    if st.button("Apply Manual & Rename to Video"):
        if not st.session_state.video_manual_boxes:
            st.error("No manual seeds")
        else:
            with st.spinner("Rendering refined video…"):
                resp = requests.post(
                    VIDEO_MANUAL_URL,
                    files={"file": (upfile.name, data, mime)},
                    data={
                        "yolo_conf": 0.5,
                        "label_map": json.dumps(st.session_state.label_map),
                        "manual_boxes": json.dumps(st.session_state.video_manual_boxes)
                    },
                )
                resp.raise_for_status()
                st.session_state.final_vid = resp.content
            st.video(st.session_state.final_vid)
            st.download_button(
                "Download Refined Video",
                data=st.session_state.final_vid,
                file_name="refined_annotated.mp4",
                mime="video/mp4"
            )

else:
    st.error("Unsupported file type.")
