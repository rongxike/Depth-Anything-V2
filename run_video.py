import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def parse_ratio(ratio_text):
    try:
        w_text, h_text = ratio_text.split(':')
        ratio_w, ratio_h = int(w_text), int(h_text)
        if ratio_w <= 0 or ratio_h <= 0:
            raise ValueError
    except Exception as exc:
        raise ValueError(f'Invalid ratio {ratio_text!r}. Use format W:H, e.g. 3:4.') from exc
    return ratio_w, ratio_h


def center_crop_to_ratio(frame, ratio_w, ratio_h):
    h, w = frame.shape[:2]
    target_ratio = ratio_w / ratio_h
    src_ratio = w / h

    if src_ratio > target_ratio:
        new_w = int(round(h * target_ratio))
        x0 = max((w - new_w) // 2, 0)
        return frame[:, x0:x0 + new_w]
    if src_ratio < target_ratio:
        new_h = int(round(w / target_ratio))
        y0 = max((h - new_h) // 2, 0)
        return frame[y0:y0 + new_h, :]
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--side-ratio', type=str, default='3:4', help='aspect ratio for each panel, format W:H')
    parser.add_argument('--margin-width', type=int, default=0, help='separator width between original and depth panels')
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()

    if args.margin_width < 0:
        parser.error('--margin-width must be >= 0')

    try:
        ratio_w, ratio_h = parse_ratio(args.side_ratio)
    except ValueError as err:
        parser.error(str(err))
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    checkpoint_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    if not os.path.isfile(checkpoint_path):
        parser.error(
            f'Missing checkpoint: {checkpoint_path}. Download the {args.encoder} weights or run with an encoder '
            'that matches an existing checkpoint.'
        )
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                filenames = [line for line in f.read().splitlines() if line]
        else:
            filenames = [args.video_path]
    else:
        filenames = [
            path for path in glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
            if os.path.isfile(path)
        ]

    if not filenames:
        parser.error(f'No video files found for --video-path {args.video_path!r}.')
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = args.margin_width
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

        if frame_width <= 0 or frame_height <= 0:
            print(f'Skipping invalid video stream: {filename}')
            raw_video.release()
            continue

        if frame_rate <= 0:
            frame_rate = 30

        if (frame_width / frame_height) > (ratio_w / ratio_h):
            side_height = frame_height
            side_width = int(round(frame_height * ratio_w / ratio_h))
        else:
            side_width = frame_width
            side_height = int(round(frame_width * ratio_h / ratio_w))
        
        if args.pred_only:
            output_width = side_width
            output_height = side_height
        else: 
            output_width = side_width * 2 + margin_width
            output_height = side_height
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, output_height))
        
        while raw_video.isOpened():
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            depth = depth_anything.infer_image(raw_frame, args.input_size)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            # Keep both panels at exactly the requested aspect ratio via centered crop.
            raw_panel = center_crop_to_ratio(raw_frame, ratio_w, ratio_h)
            depth_panel = center_crop_to_ratio(depth, ratio_w, ratio_h)
            
            if args.pred_only:
                out.write(depth_panel)
            else:
                frames = [raw_panel]
                if margin_width > 0:
                    split_region = np.ones((side_height, margin_width, 3), dtype=np.uint8) * 255
                    frames.append(split_region)
                frames.append(depth_panel)
                combined_frame = cv2.hconcat(frames)
                
                out.write(combined_frame)
        
        raw_video.release()
        out.release()
