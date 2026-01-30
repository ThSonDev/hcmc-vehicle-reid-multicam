import cv2
import numpy as np
import json
import reid_utils as utils
from confluent_kafka import Consumer

# --- CONFIG ---
BROKER = "localhost:9092"
TOPIC_MATCH = "reid_matches"
GROUP_ID = "visualizer_bottom_v4" # Đổi group ID

# --- UI CONSTANTS ---
WIN_W, WIN_H = 1000, 700
ROW_H = 150
BTN_H = 50
SCROLL_SPEED = 30

# State
matches_data = []  # Pane 1
new_cam3_data = [] # Pane 2

# Map tìm nhanh: {cam1_id: index}
cam1_id_map = {}

current_pane = 1 
scroll_y = 0

def draw_button(canvas, x, y, w, h, text, active):
    color = (0, 180, 0) if active else (80, 80, 80)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), color, -1)
    cv2.putText(canvas, text, (x + 20, y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

def resize_h(img, target_h):
    if img is None: return np.zeros((target_h, 100, 3), dtype=np.uint8)
    h, w = img.shape[:2]
    if h == 0: return np.zeros((target_h, 100, 3), dtype=np.uint8)
    scale = target_h / h
    return cv2.resize(img, (int(w*scale), target_h))

def overlay_image(background, foreground, x_offset, y_offset):
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    x1 = max(0, x_offset)
    y1 = max(0, y_offset)
    x2 = min(bg_w, x_offset + fg_w)
    y2 = min(bg_h, y_offset + fg_h)
    if x1 >= x2 or y1 >= y2: return
    fg_x1 = x1 - x_offset
    fg_y1 = y1 - y_offset
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)
    try: background[y1:y2, x1:x2] = foreground[fg_y1:fg_y2, fg_x1:fg_x2]
    except: pass

def render_ui():
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    
    t1 = f"PANE 1: TRACKING ({len(matches_data)})"
    t2 = f"PANE 2: NEW IN CAM3 ({len(new_cam3_data)})"
    draw_button(canvas, 0, 0, WIN_W//2, BTN_H, t1, current_pane==1)
    draw_button(canvas, WIN_W//2, 0, WIN_W//2, BTN_H, t2, current_pane==2)
    
    view_h = WIN_H - BTN_H
    viewport = np.zeros((view_h, WIN_W, 3), dtype=np.uint8)
    
    content_list = matches_data if current_pane == 1 else new_cam3_data
    total_rows = len(content_list)
    
    if total_rows == 0:
        cv2.putText(viewport, "WAITING FOR DATA...", (WIN_W//2 - 150, view_h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)
    else:
        start_idx = max(0, scroll_y // ROW_H)
        end_idx = min(total_rows, start_idx + (view_h // ROW_H) + 2)
        y_draw = -(scroll_y % ROW_H) 
        
        for i in range(start_idx, end_idx):
            item = content_list[i]
            
            if i % 2 != 0:
                cv2.rectangle(viewport, (0, y_draw), (WIN_W, y_draw+ROW_H), (30,30,30), -1)
            
            if current_pane == 1:
                # --- PANE 1 ---
                img1 = resize_h(utils.decode_image_base64(item.get('cam1_b64')), ROW_H-20)
                overlay_image(viewport, img1, 10, y_draw+10)
                cv2.putText(viewport, f"C1 ID: {item.get('cam1_id')}", (10, y_draw+20), 0, 0.5, (0,255,255), 1)
                cv2.putText(viewport, "->", (240, y_draw+ROW_H//2), 0, 1, (100,100,100), 2)
                
                # Cam 2
                if 'cam2_b64' in item:
                    img2 = resize_h(utils.decode_image_base64(item['cam2_b64']), ROW_H-20)
                    overlay_image(viewport, img2, 300, y_draw+10)
                    cv2.putText(viewport, f"C2 ID: {item['cam2_id']}", (300, y_draw+20), 0, 0.5, (0,255,0), 1)
                    score = item.get('cam2_score', 0)
                    cv2.putText(viewport, f"{score:.2f}", (300, y_draw+ROW_H-10), 0, 0.5, (200,200,200), 1)
                else:
                    cv2.putText(viewport, "...", (300, y_draw+ROW_H//2), 0, 1, (50,50,50), 1)
                cv2.putText(viewport, "->", (530, y_draw+ROW_H//2), 0, 1, (100,100,100), 2)
                
                # Cam 3
                if 'cam3_b64' in item:
                    img3 = resize_h(utils.decode_image_base64(item['cam3_b64']), ROW_H-20)
                    overlay_image(viewport, img3, 600, y_draw+10)
                    cv2.putText(viewport, f"C3 ID: {item['cam3_id']}", (600, y_draw+20), 0, 0.5, (0,255,0), 1)
                    score = item.get('cam3_score', 0)
                    cv2.putText(viewport, f"{score:.2f}", (600, y_draw+ROW_H-10), 0, 0.5, (200,200,200), 1)
                else:
                    cv2.putText(viewport, "...", (600, y_draw+ROW_H//2), 0, 1, (50,50,50), 1)
            else:
                # --- PANE 2 ---
                b64 = item.get('match_b64')
                if b64:
                    img = resize_h(utils.decode_image_base64(b64), ROW_H-20)
                    overlay_image(viewport, img, 50, y_draw+10)
                cv2.putText(viewport, f"NEW C3 ID: {item.get('match_id')}", (50, y_draw+20), 0, 0.6, (0,0,255), 2)
                ts = item.get('timestamp', 0)
                cv2.putText(viewport, f"TS: {ts:.2f}", (250, y_draw+ROW_H//2), 0, 0.6, (200,200,200), 1)

            cv2.line(viewport, (0, y_draw+ROW_H), (WIN_W, y_draw+ROW_H), (50,50,50), 1)
            y_draw += ROW_H

    canvas[BTN_H:, :] = viewport
    return canvas

def mouse_callback(event, x, y, flags, param):
    global current_pane, scroll_y
    if event == cv2.EVENT_LBUTTONDOWN:
        if y < BTN_H:
            current_pane = 1 if x < WIN_W//2 else 2
            scroll_y = 0

def run_visualizer():
    global scroll_y, cam1_id_map, matches_data, new_cam3_data
    consumer = utils.get_kafka_consumer(BROKER, GROUP_ID, [TOPIC_MATCH])
    
    cv2.namedWindow("ReID Visualizer")
    cv2.setMouseCallback("ReID Visualizer", mouse_callback)

    print("[Visualizer] Waiting...")

    while True:
        msg = consumer.poll(0.05)
        
        if msg and not msg.error():
            try:
                data = json.loads(msg.value().decode())
                is_new = data.get('is_new', False)
                src = data.get('cam_source')
                
                # --- LOGIC PANE 2 (New Cam 3) ---
                if is_new and src == 'cam3':
                    # Kiểm tra trùng lặp ID trước khi thêm
                    c3_id = data.get('match_id')
                    exists = any(item['match_id'] == c3_id for item in new_cam3_data)
                    if not exists:
                        # [CHANGE] Thêm vào CUỐI (Bottom)
                        new_cam3_data.append(data)
                        if len(new_cam3_data) > 100: new_cam3_data.pop(0)
                        print(f"[UI] Appended New Cam3 ID: {c3_id}")

                # --- LOGIC PANE 1 (Matched) ---
                elif not is_new:
                    # [LOGIC QUAN TRỌNG] Nếu Cam 3 match, kiểm tra xem xe này có đang ở Pane 2 không
                    if src == 'cam3':
                        c3_match_id = data.get('match_id')
                        # Lọc bỏ xe này khỏi danh sách New (nếu có)
                        initial_len = len(new_cam3_data)
                        new_cam3_data = [item for item in new_cam3_data if item.get('match_id') != c3_match_id]
                        if len(new_cam3_data) < initial_len:
                            print(f"[UI] ♻️ Moved Cam3 ID {c3_match_id} from NEW -> MATCHED")

                    c1_id = data.get('cam1_id')
                    if c1_id is not None:
                        c1_id = int(c1_id)
                        
                        # Nếu ID Cam 1 chưa có -> Tạo mới
                        if c1_id not in cam1_id_map:
                            row = { 'cam1_id': c1_id, 'cam1_b64': data.get('cam1_b64') }
                            # [CHANGE] Thêm vào CUỐI (Bottom)
                            matches_data.append(row)
                            
                            # Cập nhật map: Vì append nên index là phần tử cuối
                            cam1_id_map[c1_id] = len(matches_data) - 1
                            print(f"[UI] Appended Match Row C1 ID: {c1_id}")
                        
                        idx = cam1_id_map[c1_id]
                        # Fix ảnh Cam 1 nếu lúc trước bị thiếu
                        if matches_data[idx].get('cam1_b64') is None and data.get('cam1_b64'):
                            matches_data[idx]['cam1_b64'] = data.get('cam1_b64')

                        if src == 'cam2':
                            matches_data[idx]['cam2_id'] = data.get('match_id')
                            matches_data[idx]['cam2_b64'] = data.get('match_b64')
                            matches_data[idx]['cam2_score'] = data.get('score')
                        elif src == 'cam3':
                            matches_data[idx]['cam3_id'] = data.get('match_id')
                            matches_data[idx]['cam3_b64'] = data.get('match_b64')
                            matches_data[idx]['cam3_score'] = data.get('score')

            except Exception as e:
                print(f"[ERROR] {e}")

        img = render_ui()
        cv2.imshow("ReID Visualizer", img)
        
        k = cv2.waitKey(10)
        if k == ord('q'): break
        if k == ord('w'): scroll_y = max(0, scroll_y - SCROLL_SPEED)
        if k == ord('s'): 
            c_len = len(matches_data) if current_pane == 1 else len(new_cam3_data)
            max_s = max(0, c_len * ROW_H - (WIN_H-BTN_H))
            scroll_y = min(max_s, scroll_y + SCROLL_SPEED)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_visualizer()