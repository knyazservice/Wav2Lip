import os
import subprocess
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
from batch_face import RetinaFace

from . import audio
from .errors import (
    AudioProcessingError,
    FaceDetectionError,
    ModelLoadError,
    VideoProcessingError,
    Wav2LipException,
)
from .models import Wav2Lip


class Wav2LipProcessor:
    """
    Класс для синхронизации движения губ на видео с аудио.
    Внутренне использует тот же процесс, что и в inference.py
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/wav2lip_gan.pth",
        img_size: int = 96,
        face_det_batch_size: int = 64 * 8,
        wav2lip_batch_size: int = 128,
        out_height: int = 480,
        pads: list = None,
        nosmooth: bool = True,
        rotate: bool = False,
        crop: list = None,
        box: list = None,
        static: bool = False,
        fps: float = 25.0,
    ):
        """
        Инициализация процессора

        :param checkpoint_path: Путь к весам модели
        :param img_size: Размер изображения для обработки
        :param face_det_batch_size: Размер батча для детекции лиц
        :param wav2lip_batch_size: Размер батча для Wav2Lip
        :param out_height: Высота выходного видео
        :param pads: Отступы [верх, низ, лево, право]
        :param nosmooth: Отключить сглаживание детекции лиц
        :param rotate: Поворот видео на 90 градусов
        :param crop: Обрезка видео [верх, низ, лево, право]
        :param box: Фиксированная область лица [верх, низ, лево, право]
        :param static: Использовать только первый кадр
        :param fps: Кадров в секунду для статического изображения
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Параметры
        self.img_size = img_size
        self.face_det_batch_size = face_det_batch_size
        self.wav2lip_batch_size = wav2lip_batch_size
        self.out_height = out_height
        self.pads = pads or [0, 20, 0, 0]
        self.nosmooth = nosmooth
        self.rotate = rotate
        self.crop = crop or [0, -1, 0, -1]
        self.box = box or [-1, -1, -1, -1]
        self.static = static
        self.fps = fps

        # Загрузка моделей
        self.model = self._load_wav2lip_model(checkpoint_path)
        self.detector = self._load_face_detector()

        # Создание временной директории
        os.makedirs("temp", exist_ok=True)
        os.makedirs("results", exist_ok=True)

    def _load_wav2lip_model(self, checkpoint_path: str) -> Wav2Lip:
        """Загрузка модели Wav2Lip"""
        try:
            model = Wav2Lip()
            if self.device == "cuda":
                checkpoint = torch.load(checkpoint_path, weights_only=True)
            else:
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location=lambda storage, loc: storage,
                    weights_only=True,
                )
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            model.load_state_dict(new_state_dict)
            return model.to(self.device).eval()
        except Exception as e:
            raise ModelLoadError(f"Не удалось загрузить модель Wav2Lip: {str(e)}") from e

    def _load_face_detector(self) -> RetinaFace:
        """Загрузка детектора лиц RetinaFace"""
        try:
            if self.device == "cuda":
                detector = RetinaFace(
                    gpu_id=0,
                    model_path="checkpoints/mobilenet.pth",
                    network="mobilenet",
                )
            else:
                detector = RetinaFace(model_path="checkpoints/mobilenet.pth", network="mobilenet")
            return detector
        except Exception as e:
            raise ModelLoadError(f"Не удалось загрузить детектор лиц: {str(e)}") from e

    def _get_smoothened_boxes(self, boxes: np.ndarray, T: int = 5) -> np.ndarray:
        """Сглаживание координат рамок лиц"""
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T :]
            else:
                window = boxes[i : i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def _face_detect(self, images: list) -> list:
        """Детекция лиц на кадрах"""
        try:
            results = []
            pady1, pady2, padx1, padx2 = self.pads
            prev_rect = None

            for image in images:
                faces = self.detector([image])[0]
                if faces:
                    box, _, _ = faces[0]
                    rect = tuple(map(int, box))
                    prev_rect = rect
                else:
                    rect = prev_rect

                if rect is None:
                    raise FaceDetectionError("Лицо не обнаружено!")

                x1, y1, x2, y2 = rect
                y1 = max(0, y1 - pady1)
                y2 = min(image.shape[0], y2 + pady2)
                x1 = max(0, x1 - padx1)
                x2 = min(image.shape[1], x2 + padx2)

                results.append([x1, y1, x2, y2])

            boxes = np.array(results)
            if not self.nosmooth:
                boxes = self._get_smoothened_boxes(boxes)

            return [
                [image[y1:y2, x1:x2], (y1, y2, x1, x2)]
                for image, (x1, y1, x2, y2) in zip(images, boxes)
            ]
        except FaceDetectionError:
            raise
        except Exception as e:
            raise FaceDetectionError(f"Ошибка при детекции лиц: {str(e)}") from e

    def _datagen(self, frames: list, mels: list):
        """Генератор данных для модели"""
        img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if self.box[0] == -1:
            if not self.static:
                face_det_results = self._face_detect(frames)
            else:
                face_det_results = self._face_detect([frames[0]])
        else:
            y1, y2, x1, x2 = self.box
            face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

        for i, m in enumerate(mels):
            idx = 0 if self.static else i % len(frames)
            frame_to_save = frames[idx].copy()
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (self.img_size, self.img_size))

            img_batch.append(face)
            mel_batch.append(m)
            frame_batch.append(frame_to_save)
            coords_batch.append(coords)

            if len(img_batch) >= self.wav2lip_batch_size:
                img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

                img_masked = img_batch.copy()
                img_masked[:, self.img_size // 2 :] = 0

                img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
                mel_batch = np.reshape(
                    mel_batch,
                    [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1],
                )

                yield img_batch, mel_batch, frame_batch, coords_batch
                img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

        if len(img_batch) > 0:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2 :] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = np.reshape(
                mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1]
            )

            yield img_batch, mel_batch, frame_batch, coords_batch

    def process_video(
        self,
        face_path: Union[str, Path],
        audio_path: Union[str, Path],
        outfile: Optional[Union[str, Path]] = "results/result_voice.mp4",
    ) -> Path:
        """
        Обработка видео/изображения и аудио

        :param face_path: Путь к видео/изображению с лицом
        :param audio_path: Путь к аудиофайлу
        :param outfile: Путь для сохранения результата
        :return: Путь к обработанному видео
        """
        try:
            face_path = str(face_path)
            audio_path = str(audio_path)
            outfile = str(outfile)

            # Проверка существования файлов
            if not os.path.exists(face_path):
                raise VideoProcessingError(f"Файл не найден: {face_path}")
            if not os.path.exists(audio_path):
                raise AudioProcessingError(f"Файл не найден: {audio_path}")

            # Проверка на статическое изображение
            if os.path.isfile(face_path) and face_path.split(".")[-1].lower() in [
                "jpg",
                "png",
                "jpeg",
            ]:
                self.static = True
                full_frames = [cv2.imread(face_path)]
                if full_frames[0] is None:
                    raise VideoProcessingError(f"Не удалось прочитать изображение: {face_path}")
                fps = self.fps
            else:
                video_stream = cv2.VideoCapture(face_path)
                if not video_stream.isOpened():
                    raise VideoProcessingError(f"Не удалось открыть видео: {face_path}")

                fps = video_stream.get(cv2.CAP_PROP_FPS)
                full_frames = []
                while True:
                    still_reading, frame = video_stream.read()
                    if not still_reading:
                        video_stream.release()
                        break

                    # Обработка кадра
                    aspect_ratio = frame.shape[1] / frame.shape[0]
                    frame = cv2.resize(
                        frame, (int(self.out_height * aspect_ratio), self.out_height)
                    )

                    if self.rotate:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    y1, y2, x1, x2 = self.crop
                    if x2 == -1:
                        x2 = frame.shape[1]
                    if y2 == -1:
                        y2 = frame.shape[0]
                    frame = frame[y1:y2, x1:x2]

                    full_frames.append(frame)

                if not full_frames:
                    raise VideoProcessingError("Не удалось прочитать кадры из видео")

            # Обработка аудио
            try:
                if not audio_path.endswith(".wav"):
                    temp_wav = "temp/temp.wav"
                    subprocess.check_call(["ffmpeg", "-y", "-i", audio_path, temp_wav])
                    audio_path = temp_wav

                wav = audio.load_wav(audio_path, 16000)
                mel = audio.melspectrogram(wav)

                if np.isnan(mel.reshape(-1)).sum() > 0:
                    raise AudioProcessingError("Mel содержит nan! Возможно используется TTS голос?")

                # Подготовка mel-спектрограмм
                mel_step_size = 16
                mel_chunks = []
                mel_idx_multiplier = 80.0 / fps
                i = 0

                while True:
                    start_idx = int(i * mel_idx_multiplier)
                    if start_idx + mel_step_size > len(mel[0]):
                        mel_chunks.append(mel[:, len(mel[0]) - mel_step_size :])
                        break
                    mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
                    i += 1

            except subprocess.CalledProcessError as e:
                raise AudioProcessingError(f"Ошибка конвертации аудио: {str(e)}") from e
            except Exception as e:
                raise AudioProcessingError(f"Ошибка обработки аудио: {str(e)}") from e

            full_frames = full_frames[: len(mel_chunks)]

            # Генерация предсказаний
            try:
                gen = self._datagen(full_frames.copy(), mel_chunks)

                for i, (img_batch, mel_batch, frames, coords) in enumerate(gen):
                    if i == 0:
                        frame_h, frame_w = full_frames[0].shape[:-1]
                        out = cv2.VideoWriter(
                            "temp/result.avi",
                            cv2.VideoWriter_fourcc(*"DIVX"),
                            fps,
                            (frame_w, frame_h),
                        )

                    img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(
                        self.device
                    )
                    mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(
                        self.device
                    )

                    with torch.no_grad():
                        pred = self.model(mel_batch, img_batch)

                    pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                    for p, f, c in zip(pred, frames, coords):
                        y1, y2, x1, x2 = c
                        p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                        f[y1:y2, x1:x2] = p
                        out.write(f)

                out.release()

            except Exception as e:
                raise VideoProcessingError(f"Ошибка при генерации предсказаний: {str(e)}") from e

            # Объединение видео и аудио
            try:
                subprocess.check_call(
                    ["ffmpeg", "-y", "-i", "temp/result.avi", "-i", audio_path, outfile]
                )
            except subprocess.CalledProcessError as e:
                raise VideoProcessingError(f"Ошибка при создании финального видео: {str(e)}") from e

            return Path(outfile)
        except (
            ModelLoadError,
            FaceDetectionError,
            AudioProcessingError,
            VideoProcessingError,
        ):
            raise
        except Exception as e:
            raise Wav2LipException(f"Неожиданная ошибка: {str(e)}") from e
