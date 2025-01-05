class Wav2LipException(Exception):
    """Базовый класс для исключений Wav2Lip"""

    pass


class ModelLoadError(Wav2LipException):
    """Ошибка при загрузке модели"""

    pass


class FaceDetectionError(Wav2LipException):
    """Ошибка при детекции лица"""

    pass


class AudioProcessingError(Wav2LipException):
    """Ошибка при обработке аудио"""

    pass


class VideoProcessingError(Wav2LipException):
    """Ошибка при обработке видео"""

    pass
