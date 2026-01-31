# 🚀 AI Background Remover (RemBG Pipeline)

Мощный инструмент для автоматического удаления фона с **картинок** и **видео**.
Написан на Python, использует нейросети (U2Net) и обработку через FFmpeg.

## ✨ Возможности

- 🍏 Оптимизация под Apple Silicon (CoreML, Neural Engine)
- 🖼 Удаление фона с изображений
- 🎥 Обработка видео (mp4, webm, gif)
- 🔊 Сохранение оригинальной аудиодорожки
- 📁 Пакетная обработка папок с сохранением структуры
- 🛡 Защита от перезаписи исходных файлов

---

## 🛠 Установка

### 1. FFmpeg (обязательно)

**macOS**
```bash
brew install ffmpeg
```

**Windows**
Скачайте с сайта ffmpeg.org и добавьте путь к bin в PATH

**Linux**
```bash
sudo apt install ffmpeg
```

---

### 2. Установка проекта

```bash
git clone https://github.com/yourname/ai-background-remover.git
cd ai-background-remover
```

Создание виртуального окружения:

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

---

## 🚀 Использование

### Картинка

```bash
python main.py input.jpg output.png
```

### Видео MP4 с цветным фоном

```bash
python main.py video.mp4 result.mp4 --bg-color green
```

### Видео WebM с прозрачностью

```bash
python main.py video.mp4 result.webm
```

### Пакетная обработка папки

```bash
python main.py input_folder output_folder
```

---

## ⚙️ Параметры

- `--model` выбор модели сегментации  
  - u2net (по умолчанию)
  - isnet-anime
  - u2net_human_seg

- `--bg-color` цвет фона (black, white, green, blue)
- `--fast` ручное управление потоками

Список моделей:

```bash
python main.py --list-models
```

---

## 💻 Совместимость

- macOS Apple Silicon - CoreML
- Windows - CPU
- Linux - CPU
