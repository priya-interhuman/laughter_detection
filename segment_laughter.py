import os
import torch
import numpy as np
import librosa
import scipy
import tgt
from functools import partial
from tqdm import tqdm

from laughter_detection import laugh_segmenter, models, configs
from laughter_detection.utils import data_loaders, audio_utils, torch_utils

# Modified to be a callable function instead of a python script with args
def run_laughter_segmentation(
    audio_path: str,
    config_name: str = 'resnet_with_augmentation',
    threshold: float = 0.5,
    min_length: float = 0.2,
    output_dir: str | None = None,
    save_to_audio_files: bool = True,
    save_to_textgrid: bool = False
) -> list[dict]:
    """
    Runs laughter segmentation and optionally saves segments as audio/textgrid.

    Returns:
        A list of dicts with 'filename', 'start', and 'end' of each detected laugh.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")

    config = configs.CONFIG_MAP[config_name]
    model = config['model'](
        dropout_rate=0.0,
        linear_layer_size=config['linear_layer_size'],
        filter_sizes=config['filter_sizes']
    )
    model.set_device(device)
    model.eval()

    feature_fn = config['feature_fn']

    #TODO[pmsarkar] this is a HACK
    if os.path.exists("/app"):
        base_dir = "/app"
    else:
        base_dir = os.getcwd()

    model_path = os.path.join(base_dir, 'laughter_detection', 'checkpoints', 'in_use', 'resnet_with_augmentation', 'best.pth.tar')

    if os.path.exists(model_path):
        torch_utils.load_checkpoint(model_path, model)
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path,
        feature_fn=feature_fn,
        sr=8000
    )

    collate_fn = partial(
        audio_utils.pad_sequences_with_labels,
        expand_channel_dim=config['expand_channel_dim']
    )

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset,
        num_workers=0,  # Set to 0 for safety in server environments
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )

    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        preds = [float(preds)] if preds.ndim == 0 else list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(audio_path)
    fps = len(probs) / file_length
    probs = laugh_segmenter.lowpass(probs)

    instances = laugh_segmenter.get_laughter_instances(probs, threshold, min_length, fps)
    print(f"Found {len(instances)} laughs.")

    if not instances:
        return []

    full_res_y, full_res_sr = librosa.load(audio_path, sr=44100)
    maxv = np.iinfo(np.int16).max
    wav_paths = []

    if save_to_audio_files:
        if output_dir is None:
            raise ValueError("Output directory must be provided to save audio files.")
        os.makedirs(output_dir, exist_ok=True)
        for index, instance in enumerate(instances):
            laughs = laugh_segmenter.cut_laughter_segments([instance], full_res_y, full_res_sr)
            wav_path = os.path.join(output_dir, f"laugh_{index}.wav")
            scipy.io.wavfile.write(wav_path, full_res_sr, (laughs * maxv).astype(np.int16))
            wav_paths.append(wav_path)

    if save_to_textgrid:
        if output_dir is None:
            raise ValueError("Output directory must be provided to save TextGrid.")
        tg = tgt.TextGrid()
        tier = tgt.IntervalTier(
            name='laughter',
            objects=[tgt.Interval(start=i[0], end=i[1], text='laugh') for i in instances]
        )
        tg.add_tier(tier)
        fname = os.path.splitext(os.path.basename(audio_path))[0]
        textgrid_path = os.path.join(output_dir, f"{fname}_laughter.TextGrid")
        tgt.write_to_file(tg, textgrid_path)

    return [
        {
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3)
        }
        for path, (start, end) in zip(wav_paths, instances)
    ]
