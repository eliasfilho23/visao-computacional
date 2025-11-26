"""
Módulo para processamento de imagens e extração de texto usando OCR (Tesseract)
"""
import pytesseract
from PIL import Image
import io


def extract_text_from_image(image_data):
    try:
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError("Formato de imagem não suportado. Use bytes ou PIL Image.")
        
        custom_config = r'--oem 3 --psm 6'
        extracted_text = pytesseract.image_to_string(image, config=custom_config, lang='por+eng')
        
        return extracted_text.strip()
    
    except Exception as e:
        raise Exception(f"Erro ao processar imagem: {str(e)}")


def extract_text_with_confidence(image_data):
    try:
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError("Formato de imagem não suportado. Use bytes ou PIL Image.")
        
        custom_config = r'--oem 3 --psm 6'
        data = pytesseract.image_to_data(image, config=custom_config, lang='por+eng', output_type=pytesseract.Output.DICT)
        
        text = pytesseract.image_to_string(image, config=custom_config, lang='por+eng')
        
        confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'text': text.strip(),
            'confidence': round(avg_confidence, 2)
        }
    
    except Exception as e:
        raise Exception(f"Erro ao processar imagem: {str(e)}")

