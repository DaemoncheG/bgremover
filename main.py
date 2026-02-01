import sys
import os
import cv2
import argparse
import subprocess
import multiprocessing
import queue
import io
import platform
import shutil
import numpy as np
from rembg import remove, new_session
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from PIL import Image, ImageOps

# ==============================================================================
# 🏁 ENDGAME EDITION (Final)
# ==============================================================================

DEFAULT_MODEL = "u2net" 

DEFAULT_IMG_EXT = "png"
DEFAULT_VID_EXT = "webm"

IMG_QUALITY   = 95
VIDEO_BITRATE = "2M"
GIF_FPS_LIMIT = 0

AVAILABLE_MODELS = [
    "u2net",
    "u2netp",
    "u2net_human_seg",
    "u2net_cloth_seg",
    "silueta",
    "isnet-general-use",
    "isnet-anime",
    "sam",
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-portrait",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-cod",
    "birefnet-massive",
    "bria-rmbg",
]

# ==============================================================================

worker_session = None
HAS_X264 = False

IMG_FORMATS_READ = ['.png', '.webp', '.jpg', '.jpeg', '.bmp', '.tiff']
VIDEO_FORMATS_READ = ['.mp4', '.mov', '.avi', '.gif', '.mkv', '.webm']

def check_environment():
    global HAS_X264

    if not shutil.which("ffmpeg"):
        print("❌ Ошибка: FFmpeg не найден! (brew install ffmpeg / apt install ffmpeg)")
        sys.exit(1)

    # FIX: Более надежная проверка libx264 (через help encoder)
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "encoder=libx264"], 
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        HAS_X264 = True
    except:
        HAS_X264 = False
        print("⚠️  ВНИМАНИЕ: libx264 (H.264) не найден. Вывод в MP4 будет недоступен.")

    if platform.system() == 'Darwin' and platform.machine() == 'x86_64':
        try:
            is_translated = subprocess.check_output(["sysctl", "-in", "sysctl.proc_translated"]).decode().strip()
            if is_translated == "1":
                print("⚠️  ВНИМАНИЕ: Python запущен под Rosetta. Скорость снижена.")
        except:
            pass

def get_hardware_defaults():
    system = platform.system()
    machine = platform.machine()
    if system == 'Darwin' and machine == 'arm64':
        return 2, 0 
    else:
        return 0, 2 

def init_worker(role_queue, model_name, verbose):
    global worker_session
    os.environ["ORT_LOGGING_LEVEL"] = "3"
    providers = ['CPUExecutionProvider']
    if role_queue is not None:
        try:
            provider_type = role_queue.get(timeout=5)
            if provider_type == 'coreml':
                providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        except queue.Empty:
            pass 
    try:
        worker_session = new_session(model_name, providers=providers)
    except Exception as e:
        if verbose: print(f"⚠️ [Worker] Fallback to CPU. ({e})")
        worker_session = new_session(model_name, providers=['CPUExecutionProvider'])

# --- AUDIO ---

def transfer_audio(source_path, target_path, verbose=False):
    if not os.path.exists(source_path) or not os.path.exists(target_path): return
    ext = os.path.splitext(target_path)[1].lower()
    temp_output = target_path + ".temp" + ext 

    enc_audio_codec = "libopus" if ext == '.webm' else "aac"

    stderr_mode = None if verbose else subprocess.DEVNULL
    stdout_mode = subprocess.DEVNULL
    base_flags = ['-hide_banner', '-nostdin', '-y']

    # Attempt 1: Copy
    cmd_copy = ['ffmpeg'] + base_flags + [
        '-i', target_path, '-i', source_path,
        '-map', '0:v', '-map', '1:a:0?', 
        '-c:v', 'copy', '-c:a', 'copy', '-shortest',
    ]
    if ext == '.mp4': cmd_copy.extend(['-movflags', '+faststart'])
    cmd_copy.append(temp_output)

    try:
        subprocess.run(cmd_copy, check=True, stdout=stdout_mode, stderr=stderr_mode)
        os.replace(temp_output, target_path)
        return
    except subprocess.CalledProcessError:
        if os.path.exists(temp_output): 
            try: os.remove(temp_output)
            except: pass

    # Attempt 2: Encode
    cmd_enc = ['ffmpeg'] + base_flags + [
        '-i', target_path, '-i', source_path,
        '-map', '0:v', '-map', '1:a:0?',
        '-c:v', 'copy', '-c:a', enc_audio_codec, '-shortest',
    ]
    if ext == '.mp4': cmd_enc.extend(['-movflags', '+faststart'])
    cmd_enc.append(temp_output)

    try:
        subprocess.run(cmd_enc, check=True, stdout=stdout_mode, stderr=stderr_mode)
        if os.path.exists(temp_output): os.replace(temp_output, target_path)
    except Exception:
        if os.path.exists(temp_output): 
            try: os.remove(temp_output)
            except: pass

# --- PROCESSORS ---

def process_frame_safe(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    result_pil = remove(pil_img, session=worker_session).convert("RGBA")
    return np.array(result_pil, dtype=np.uint8)

def process_single_image_task(task):
    input_path, output_path = task
    try:
        with open(input_path, 'rb') as i: input_data = i.read()
        img_pil = Image.open(io.BytesIO(input_data))
        img_pil = ImageOps.exif_transpose(img_pil).convert("RGBA")
        out_pil = remove(img_pil, session=worker_session).convert("RGBA")

        if output_path.lower().endswith(('.jpg', '.jpeg')):
            bg = Image.new("RGB", out_pil.size, (255, 255, 255))
            bg.paste(out_pil, mask=out_pil.split()[3])
            bg.save(output_path, quality=IMG_QUALITY)
        elif output_path.lower().endswith('.webp'):
            out_pil.save(output_path, format='WEBP', quality=IMG_QUALITY, method=6)
        else:
            out_pil.save(output_path)
        return True
    except Exception as e:
        print(f"❌ {os.path.basename(input_path)}: {e}")
        return False

# --- HELPERS ---

def get_total_workers(args): return max(1, args.coreml + args.cpu)
def prepare_roles(manager, args):
    q = manager.Queue()
    for _ in range(args.coreml): q.put('coreml')
    for _ in range(args.cpu): q.put('cpu')
    return q
def is_video(p): return p.lower().endswith(tuple(VIDEO_FORMATS_READ))
def is_image(p): return p.lower().endswith(tuple(IMG_FORMATS_READ))

def norm_path(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))

def is_same_path(p1, p2):
    return norm_path(p1) == norm_path(p2)

def force_ext(path, wanted_ext):
    base, _ = os.path.splitext(path)
    clean_ext = wanted_ext.lstrip('.')
    return f"{base}.{clean_ext}"

def safe_remove(path):
    """Безопасное удаление файла (не падает на Windows Lock)"""
    try:
        if os.path.exists(path): os.remove(path)
    except Exception as e:
        print(f"⚠️ Не удалось удалить временный файл {path}: {e}")

# --- PIPELINES ---

def process_images_batch(files_list, args, manager):
    total_workers = get_total_workers(args)
    role_queue = prepare_roles(manager, args)
    print(f"🖼 Картинки: {len(files_list)} шт.")
    with ProcessPoolExecutor(max_workers=total_workers, initializer=init_worker, initargs=(role_queue, args.model, args.verbose)) as executor:
        # FIX: Memory Optimization (Streaming instead of list)
        for _ in tqdm(executor.map(process_single_image_task, files_list), total=len(files_list), unit="img"): pass

def process_video_mp4_h264(input_path, output_path, args, role_queue, bg_color):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not total or total <= 0: total = None

        print(f"🎥 MP4 (H.264, Bg: {bg_color}): {os.path.basename(input_path)}")

        cmd = [
            'ffmpeg', '-hide_banner', '-nostdin', '-y', 
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'bgr24',
            '-r', str(fps), '-i', '-', 
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '23',
            '-movflags', '+faststart',
            output_path
        ]

        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=None if args.verbose else subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        ffmpeg_dead = False

        try:
            bg_static = np.zeros((h, w, 3), dtype=np.uint8)
            bg_static[:] = bg_color
            bg_static_f = bg_static.astype(np.float32)

            with ProcessPoolExecutor(max_workers=get_total_workers(args), initializer=init_worker, initargs=(role_queue, args.model, args.verbose)) as executor:
                bs = get_total_workers(args) * 2
                with tqdm(total=total, unit="frame", leave=False) as pbar:
                    while cap.isOpened():
                        if proc.poll() is not None: 
                            ffmpeg_dead = True
                            break

                        batch = []
                        for _ in range(bs):
                            ret, frame = cap.read()
                            if not ret: break
                            batch.append(frame)
                        if not batch: break

                        # FIX: Memory Optimization (Streaming iterator)
                        for rgba in executor.map(process_frame_safe, batch):
                            alpha = (rgba[..., 3:4].astype(np.float32) / 255.0)
                            fg_bgr = rgba[..., :3][:, :, ::-1].astype(np.float32)
                            final_float = (fg_bgr * alpha + bg_static_f * (1 - alpha))
                            final_uint8 = final_float.astype(np.uint8)

                            try: proc.stdin.write(np.ascontiguousarray(final_uint8).tobytes())
                            except (BrokenPipeError, OSError): 
                                ffmpeg_dead = True
                                break

                        if ffmpeg_dead: break
                        pbar.update(len(batch))
        finally:
            if proc.stdin: proc.stdin.close()
            proc.wait()

        if proc.returncode == 0: 
            transfer_audio(input_path, output_path, args.verbose)
        else:
            print(f"❌ Ошибка: FFmpeg завершился с кодом {proc.returncode}")
            safe_remove(output_path) # FIX: Safe remove

    finally: cap.release()

def process_webm_transparent(input_path, output_path, args, role_queue):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not total or total <= 0: total = None
        print(f"💎 WebM: {os.path.basename(input_path)}")

        cmd = [
            'ffmpeg', '-hide_banner', '-nostdin', '-y', 
            '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{w}x{h}', '-pix_fmt', 'rgba',
            '-r', str(fps), '-i', '-', '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p',
            '-b:v', VIDEO_BITRATE, '-auto-alt-ref', '0', output_path
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=None if args.verbose else subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        ffmpeg_dead = False

        with ProcessPoolExecutor(max_workers=get_total_workers(args), initializer=init_worker, initargs=(role_queue, args.model, args.verbose)) as executor:
            bs = get_total_workers(args)
            with tqdm(total=total, unit="frame", leave=False) as pbar:
                while cap.isOpened():
                    if proc.poll() is not None: 
                        ffmpeg_dead = True
                        break

                    batch = []
                    for _ in range(bs):
                        ret, frame = cap.read()
                        if not ret: break
                        batch.append(frame)
                    if not batch: break

                    # FIX: Memory Optimization (Streaming iterator)
                    for frame_rgba in executor.map(process_frame_safe, batch):
                        try: proc.stdin.write(np.ascontiguousarray(frame_rgba, dtype=np.uint8).tobytes())
                        except (BrokenPipeError, OSError): 
                            ffmpeg_dead = True
                            break

                    if ffmpeg_dead: break
                    pbar.update(len(batch))
        if proc.stdin: proc.stdin.close()
        proc.wait()
        if proc.returncode == 0: 
            transfer_audio(input_path, output_path, args.verbose)
        else:
            print(f"❌ Ошибка: FFmpeg завершился с кодом {proc.returncode}")
            safe_remove(output_path)
    finally: cap.release()

def process_gif_output(input_path, output_path, args, role_queue):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened(): return
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        if GIF_FPS_LIMIT > 0 and fps > GIF_FPS_LIMIT: fps = GIF_FPS_LIMIT
        duration = 1000 / fps
        print(f"🎞 GIF: {os.path.basename(input_path)}")

        frames = []
        with ProcessPoolExecutor(max_workers=get_total_workers(args), initializer=init_worker, initargs=(role_queue, args.model, args.verbose)) as executor:
            bs = get_total_workers(args) * 2
            while cap.isOpened():
                batch = []
                for _ in range(bs):
                    ret, frame = cap.read()
                    if not ret: break
                    batch.append(frame)
                if not batch: break

                # GIF все равно требует накопления в RAM, тут стриминг не поможет без ffmpeg pipe
                for res in executor.map(process_frame_safe, batch):
                    frames.append(Image.fromarray(res))

        if frames: frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0, disposal=2)
    finally: cap.release()

def parse_bg_color(name):
    return {'black':(0,0,0), 'white':(255,255,255), 'green':(0,255,0), 'blue':(255,0,0)}.get(name.lower(), (0,0,0))

def run_processing(input_path, output_path, args, manager):
    in_abs = norm_path(input_path)
    out_abs = norm_path(output_path)

    if os.path.isdir(in_abs):
        # FIX: UX - проверяем точку в имени папки
        if os.path.splitext(output_path)[1] and not os.path.isdir(output_path) and not output_path.endswith(os.sep):
             print("❌ Ошибка: При пакетной обработке выходной путь должен быть папкой.")
             print("   Совет: Если папка содержит точку (напр. v1.0), добавьте слэш в конце: 'v1.0/'")
             return

        try:
            if os.path.commonpath([in_abs, out_abs]) == in_abs:
                print("❌ Ошибка: Папка вывода находится внутри папки ввода!")
                return
        except ValueError:
            pass 

    files_img = []
    files_vid = []

    if os.path.isfile(input_path):
        is_folder_intent = os.path.isdir(output_path) or output_path.endswith(os.sep) or not os.path.splitext(output_path)[1]

        if is_image(input_path):
            target_ext = args.img_format
            if is_folder_intent:
                fname = os.path.splitext(os.path.basename(input_path))[0]
                final_out = os.path.join(output_path, f"{fname}.{target_ext}")
            else:
                final_out = force_ext(output_path, target_ext)

            if os.path.splitext(output_path)[1].lower() != os.path.splitext(final_out)[1].lower() and not is_folder_intent:
                 print(f"⚠️ Расширение изменено: {output_path} -> {final_out}")

            if is_same_path(input_path, final_out):
                print(f"❌ Пропуск: Вход == Выход ({final_out})")
                return
            files_img.append((input_path, final_out))

        elif is_video(input_path):
            target_ext = args.vid_format
            if is_folder_intent:
                fname = os.path.splitext(os.path.basename(input_path))[0]
                final_out = os.path.join(output_path, f"{fname}.{target_ext}")
            else:
                final_out = force_ext(output_path, target_ext)

            if os.path.splitext(output_path)[1].lower() != os.path.splitext(final_out)[1].lower() and not is_folder_intent:
                 print(f"⚠️ Расширение изменено: {output_path} -> {final_out}")

            if is_same_path(input_path, final_out):
                print(f"❌ Пропуск: Вход == Выход ({final_out})")
                return
            files_vid.append((input_path, final_out))
    else:
        for root, dirs, files in os.walk(input_path):
            for file in files:
                fpath = os.path.join(root, file)
                rel = os.path.relpath(root, input_path)
                dest_fold = os.path.join(output_path, rel)
                fname = os.path.splitext(file)[0]

                if is_image(fpath):
                    files_img.append((fpath, os.path.join(dest_fold, fname + f".{args.img_format}")))
                elif is_video(fpath):
                    files_vid.append((fpath, os.path.join(dest_fold, fname + f".{args.vid_format}")))

    for _, out in files_img + files_vid:
        d = os.path.dirname(out)
        if d: os.makedirs(d, exist_ok=True)

    if files_img: process_images_batch(files_img, args, manager)

    bg = parse_bg_color(args.bg_color)
    for i, (inp, out) in enumerate(files_vid):
        print(f"[{i+1}/{len(files_vid)}] Видео...")
        q = prepare_roles(manager, args)
        if out.lower().endswith('.gif'): process_gif_output(inp, out, args, q)
        elif out.lower().endswith('.webm'): process_webm_transparent(inp, out, args, q)
        elif args.vid_format == 'mp4' and not HAS_X264:
             print(f"❌ Пропуск {inp}: Нет кодека libx264 для MP4.")
        else: process_video_mp4_h264(inp, out, args, q, bg)
    print("\n✨ Готово!")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    check_environment()
    def_coreml, def_cpu = get_hardware_defaults()

    parser = argparse.ArgumentParser(description="RemBG Pipeline (Endgame Edition)", formatter_class=argparse.RawTextHelpFormatter, epilog="Models: "+", ".join(AVAILABLE_MODELS))
    parser.add_argument("input", nargs='?', help="Input")
    parser.add_argument("output", nargs='?', help="Output")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, choices=AVAILABLE_MODELS)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--img-format", default=DEFAULT_IMG_EXT, choices=["png", "webp", "jpg"])
    parser.add_argument("--vid-format", default=DEFAULT_VID_EXT, choices=["webm", "mp4", "gif"])
    parser.add_argument("--coreml", type=int, default=def_coreml)
    parser.add_argument("--cpu", type=int, default=def_cpu)
    parser.add_argument("--bg-color", default="black")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if args.list_models:
        print("\n".join(AVAILABLE_MODELS))
        sys.exit(0)

    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)
    if not os.path.exists(args.input):
        print("❌ Вход не найден")
        sys.exit(1)

    with multiprocessing.Manager() as manager:
        run_processing(args.input, args.output, args, manager)
