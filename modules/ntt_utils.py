
import logging
log = logging.getLogger(__name__)

import io
import os
import sys
import time
import numpy as np
import subprocess
import requests
import torch
import whisper
import PyPDF2
import fitz  # PyMuPDF
from PIL import Image
from pdf2image import convert_from_path

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

two_lang_code_to_three = {
    "af": "afr",
    "am": "amh",
    "ar": "ara",
    "as": "asm",
    "az": "aze",
    "be": "bel",
    "bg": "bul",
    "bn": "ben",
    "bo": "bod",
    "br": "bre",
    "bs": "bos",
    "ca": "cat",
    "cs": "ces",
    "cy": "cym",
    "da": "dan",
    "de": "deu",
    "el": "ell",
    "en": "eng",
    "es": "spa",
    "et": "est",
    "eu": "eus",
    "fa": "fas",
    "fi": "fin",
    "fo": "fao",
    "fr": "fra",
    "gl": "glg",
    "gu": "guj",
    "ha": "hau",
    "he": "heb",
    "hi": "hin",
    "hr": "hrv",
    "hu": "hun",
    "hy": "hye",
    "id": "ind",
    "is": "isl",
    "it": "ita",
    "ja": "jpn",
    "jw": "jav",
    "ka": "kat",
    "kk": "kaz",
    "km": "khm",
    "kn": "kan",
    "ko": "kor",
    "la": "lat",
    "lb": "ltz",
    "ln": "lin",
    "lo": "lao",
    "lt": "lit",
    "lv": "lav",
    "mg": "mlg",
    "mi": "mri",
    "mk": "mkd",
    "ml": "mal",
    "mn": "mon",
    "mr": "mar",
    "ms": "msa",
    "mt": "mlt",
    "my": "mya",
    "ne": "nep",
    "nl": "nld",
    "no": "nor",
    "oc": "oci",
    "pa": "pan",
    "pl": "pol",
    "ps": "pus",
    "pt": "por",
    "ro": "ron",
    "ru": "rus",
    "sa": "san",
    "sd": "snd",
    "si": "sin",
    "sk": "slk",
    "sl": "slv",
    "sn": "sna",
    "so": "som",
    "sq": "sqi",
    "sr": "srp",
    "su": "sun",
    "sv": "swe",
    "sw": "swa",
    "ta": "tam",
    "te": "tel",
    "tg": "tgk",
    "th": "tha",
    "tk": "tuk",
    "tl": "tgl",
    "tr": "tur",
    "tt": "tat",
    "ug": "uig",
    "uk": "ukr",
    "ur": "urd",
    "uz": "uzb",
    "vi": "vie",
    "yi": "yid",
    "yo": "yor"
    }

whisper_model = None

class NTTUtils:

    def print_info():
        log.info(f"python path: {sys.executable}")
        log.info(f"numpy version: {np.__version__}")
        if torch.cuda.is_available():
            log.info(f"cuda device count: {torch.cuda.device_count()}")
            log.info(f"cuda current device: {torch.cuda.current_device()}")
            log.info(f"device name: {torch.cuda.get_device_name(0)}")

    def init_whisper(model="tiny"):
        #install ffmpeg
        #install latest nvidia driver
        #install cuda 12.6
        #run nvcc --version
        #pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
        #pip install git+https://github.com/openai/whisper.git
        #trouble shooting
        #if import torch fail with "The specified module could not be found. ...\fbgemm.dll or one of its dependencies. 
        #copy libomp140.x86_64.dll to where fbgemm.dll installed
        #if Numy is not availablepip uninstall numpy | pip install numpy==1.26.4

        #Load the model (you can specify "tiny", "base", "small", "medium", or "large")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"loading whisper model: {model}...")
        NTTUtils.whisper_model = whisper.load_model(model).to(device)
        log.info(f"loading whisper model: {model} done.")

    def transcribe(file_path,language="en"):

        if NTTUtils.whisper_model is None:
            log.error("The whisper_model is None. call init_whisper")
            return {}
        
        file_type = NTTUtils.get_file_type(file_path)
        if file_type not in ("audio", "video"):
            log.error("only audio and video are accepted for transcription",file_path)
            return {}
        
        start_time = time.time()
        duration = NTTUtils.get_media_duration(file_path)

        log.debug(f"transcribing... {file_path} using language: {language} {duration:.2f} seconds")
        result = NTTUtils.whisper_model.transcribe(file_path, language=language)
        elapsed_time = time.time() - start_time
        log.debug(f"transcribing done. {file_path} using language: {language} {duration:.2f} seconds , took: {elapsed_time:.2f} seconds.")

        srt_lines = []
        for i, segment in enumerate(result["segments"], 1):
            stime = NTTUtils.format_timestamp(segment["start"])
            etime = NTTUtils.format_timestamp(segment["end"])
            text = segment["text"].strip()
            srt_lines.extend([
                str(i),
                f"{stime} --> {etime}",
                text + "\n"
            ])
            result["srt"] = "\n".join(srt_lines)

        return result

    def transcription_to_srt(result):
        return

    def summarize(text, language, model, host):
        start_time = time.time()
        log.debug(f"summarizing... {len(text)} chars => {language}")
    
        prompts = {
            "en": f"""Please provide a concise summary of the following text:

                    {text}

                    Summary:""",
            
            "he": f"""אנא ספק תקציר תמציתי של הטקסט הבא:

                    {text}

                    תקציר:""",
            
            "ru": f"""Предоставьте краткое содержание следующего текста:

                    {text}

                    Краткое содержание:"""
        }
        
        # Get appropriate prompt template
        prompt = prompts.get(language, prompts[language])
        
        # Request payload with UTF-8 encoding
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512
            }
        }
        
        try:
            # Make request with proper encoding
            headers = {"Content-Type": "application/json; charset=utf-8"}
            response = requests.post(url=f"{host}/api/generate", 
                                json=payload,
                                headers=headers)
            response.raise_for_status()
            
            result = response.json()
        
            elapsed_time = time.time() - start_time
            log.debug(f"summarizing done. {len(text)} chars => {len(result['response'])} language={language} , took: {elapsed_time:.2f} seconds.")

            return result["response"]
            
        except requests.exceptions.RequestException as e:
            log.error(f"Error making request to Ollama: {e}")
            return "fail to summarize"

    def image_to_ocr_text(file_path,language="en"):
        start_time = time.time()
        log.debug(f"image_to_text... {file_path} language: {language}")
        lang = two_lang_code_to_three[language]
        if lang!="eng":
            lang = lang + "+eng"
        try:
            img = Image.open(file_path)
            text = pytesseract.image_to_string(img, lang=lang) 
            elapsed_time = time.time() - start_time
            log.debug(f"image_to_text done. {file_path} language: {language}, took: {elapsed_time:.2f} seconds.")
            return text
        except FileNotFoundError:
            log.error(f"image_to_text Image file not found at {file_path}")
            return None
        except Exception as e:
            log.error(f"image_to_text: {e}")
            return None

    def pdf_to_ocr_text(file_path,language="en"):
        start_time = time.time()
        utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
        poppler_path = os.path.join(utils_path, "PopplerRelease-24.08.0-0","Library","bin")

        log.debug(f"ocr_pdf... {file_path} language: {language}")
        lang = two_lang_code_to_three[language]
        if lang!="eng":
            lang = lang + "+eng"
        try:
            doc = fitz.open(file_path)  # Open PDF with PyMuPDF
            num_pages = doc.page_count
            #images = convert_from_path(file_path,poppler_path=poppler_path) #gets image per page array
        except Exception as e: # Catch potential PDF loading errors
            log.error(f"Error extracting PDF num_pages: {e}")
            return []

        result = []

        for page_num in range(num_pages):
            page = doc[page_num]
            text_from_pdf = page.get_text("text")  # Attempt text extraction first
            ocr_text = ""

            log.debug(f"ocr_pdf ... {file_path} language: {language} page: {page_num} of {num_pages} ...")

            try:
                images = convert_from_path(file_path, first_page=page_num+1, last_page=page_num+1, poppler_path=settings["poppler_path"])

                if images: # Check if images were generated successfully
                    image = images[0]
                    image_bytes = io.BytesIO()
                    image.save(image_bytes, format="PNG")
                    image_bytes.seek(0)
                    img = Image.open(image_bytes)
                    ocr_text = pytesseract.image_to_string(img, lang=lang)
                else:
                    log.debug(f"No image generated for page {page_num+1}")

            except Exception as e:
                log.error(f"OCR Error on page {page_num+1}: {e}")

            result.append({"text": text_from_pdf, "ocr": ocr_text})

        elapsed_time = time.time() - start_time
        log.debug(f"ocr_pdf done. {file_path} language: {language}, took: {elapsed_time:.2f} seconds.")
        return result

    def pdf_to_text(file_path):
        try:
            with open(file_path, 'rb') as file:
                
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"

                return text.strip()
            
        except Exception as e:
            log.error(f"Error processing PDF {file_path}: {e}")
            return ""
    
    def pdf_to_text_with_image_to_ocr_text(pdf_path):
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images(full=True)
            page_text = page.get_text() # Extract any text that's already there
            full_text += page_text

            for image_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                full_text += ocr_text

        return full_text
     
    def get_file_type(file_path):
        """Determines the file type based on its extension."""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()[1:]  # Remove the leading dot
        if ext in ["txt"]:
            return "text"
        if ext in ["doc","docx"]:
            return "doc"
        if ext in ["xlsx","csv"]:
            return "excel"
        if ext in ["jpg", "jpeg", "png", "bmp", "gif", "tiff"]:
            return "image"
        elif ext == "pdf":
            return "pdf"
        elif ext in ["mp3", "wav", "flac", "ogg"]: #common audio extensions
            return "audio"
        elif ext in ["mp4", "avi", "mov", "mkv", "ts"]: #common video extensions
            return "video"
        else:
            return "unknown"

    def replace_extension(file_path, new_extension):
        base = os.path.splitext(file_path)[0]
        return f"{base}{new_extension}"

    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        msecs = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{msecs:03d}"
    
    def wait_for_file_ready(file_path, retries=20, wait_time=1):
        """
        Check if the file is ready for processing by trying to open it.
        
        :param file_path: The path of the file to check.
        :param retries: Number of times to attempt to open the file.
        :param wait_time: Time in seconds to wait between attempts.
        :return: True if the file is ready, False otherwise.
        """
        for _ in range(retries):
            try:
                # Try to open the file exclusively
                with open(file_path, "rb") as f:
                    f.read()
                return True  # File is ready
            except IOError:
                time.sleep(wait_time)  # Wait and try again

        return False

    def get_media_duration(file_path):

        utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils"))
        ffprobe_path = os.path.join(utils_path, "ffprobe.exe")

        try:
            result = subprocess.run(
               [
                ffprobe_path,
                "-v","error",
                "-show_entries",
                "format=duration",
                "-of","default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=False,  # Don't raise an exception automatically
            )

            if result.returncode != 0: #Check if there was an error
                log.error(f"get_media_duration ffprobe returned an error: {result.stderr}")
                return 0.0

            return float(result.stdout.strip())
        except Exception as e:
            log.error(f"get_media_duration {file_path} Error occurred: {e}") 
            return 0.0
    
