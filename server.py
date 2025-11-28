import os
import requests
import argparse
import json
import re
import base64
import numpy as np
import cv2
from PIL import Image
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from image_processor import extract_text_from_image, extract_text_with_confidence

load_dotenv('./.env')

app = Flask(__name__)
CORS(app)

GOOG_API_KEY = os.getenv('GOOG_API_KEY')
SERVER_PORT = os.getenv('SERVER_PORT')

if not GOOG_API_KEY:
    raise ValueError("GOOG_API_KEY não encontrada nas variáveis de ambiente. Por favor, configure a variável GOOG_API_KEY.")

def format_prompt(user_text):
    prompt = f"""Analise o seguinte texto extraído de uma imagem e identifique cada frase ou sentença separadamente. Para cada frase, retorne um JSON estruturado com as seguintes informações:

Texto a analisar:
{user_text}

Instruções:
1. Identifique e separe cada frase ou sentença do texto
2. Para cada frase, analise:
   - A métrica de credibilidade (valor de 1 a 5, onde 1 = muito falso, 5 = muito verdadeiro)
   - Um detalhamento explicando a análise

3. Além disso, faça uma análise geral considerando TODAS as frases em conjunto, avaliando o texto completo como um todo.

4. Retorne APENAS um JSON válido no seguinte formato (sem markdown, sem explicações adicionais):

{{
  "frase_1": {{
    "probabilidade_de_ser_meme": <número de 0 a 100>,
    "detalhamento": "<explicação detalhada da análise>"
  }},
  "frase_2": {{
    "probabilidade_de_ser_meme": <número de 0 a 100>,
    "detalhamento": "<explicação detalhada da análise>"
  }},
  "analise_geral": {{
    "probabilidade_de_ser_meme": <número de 0 a 100, considerando todas as frases em conjunto>,
    "detalhamento": "<explicação detalhada da análise geral do texto completo, considerando o contexto e a relação entre todas as frases>"
  }}
}}

IMPORTANTE: Retorne APENAS o JSON, sem markdown, sem código, sem explicações. Apenas o JSON puro."""
    return prompt

def get_validation_parameters(user_text):

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': GOOG_API_KEY
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": user_text
                    }
                ]
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response

def extract_comprehensible_text(extracted_text):
    prompt = f"""Extraia e retorne apenas o texto compreensível e legível do seguinte texto extraído de uma imagem por OCR. 
Remova caracteres estranhos, erros de reconhecimento óptico e mantenha apenas o texto que faz sentido:

{extracted_text}

Retorne apenas o texto limpo e compreensível, sem explicações adicionais."""
    
    response = get_validation_parameters(prompt)
    
    if response.status_code == 200:
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            if 'content' in result['candidates'][0]:
                parts = result['candidates'][0]['content'].get('parts', [])
                if parts and 'text' in parts[0]:
                    return parts[0]['text'].strip()
    
    return extracted_text

def extract_json_from_response(text):
    """
    Extrai JSON da resposta do Gemini, removendo markdown e texto adicional
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Tenta encontrar JSON entre chaves
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(0)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Se não encontrou, tenta parsear o texto inteiro
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def calculate_average_probability(analysis_dict):
    """
    Calcula a média de todas as probabilidades_de_ser_meme em um dicionário de análise.
    
    Args:
        analysis_dict: dicionário com análises (ex: text_analysis ou image_analysis)
        
    Returns:
        float: média das probabilidades, ou None se não houver probabilidades
    """
    if not analysis_dict or not isinstance(analysis_dict, dict):
        return None
    
    probabilities = []
    
    def extract_probabilities(obj):
        """Função recursiva para extrair todas as probabilidades"""
        if isinstance(obj, dict):
            if 'probabilidade_de_ser_meme' in obj:
                prob = obj['probabilidade_de_ser_meme']
                if isinstance(prob, (int, float)):
                    probabilities.append(prob)
            # Recursivamente processar todos os valores do dicionário
            for value in obj.values():
                extract_probabilities(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_probabilities(item)
    
    extract_probabilities(analysis_dict)
    
    if not probabilities:
        return None
    
    return sum(probabilities) / len(probabilities)

def preprocess_image(image_data):
    """
    Aplica pré-processamento na imagem (grayscale, equalização, normalização, blur).
    Baseado no código de pprocess.py linhas 9-17.
    
    Args:
        image_data: bytes da imagem
        
    Returns:
        bytes: imagem pré-processada em formato JPEG
    """
    try:
        # Converter bytes para numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Erro ao decodificar a imagem.")
        
        # Processamento da imagem em escala de cinza
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgEqualized = cv2.equalizeHist(imgGray)
        imgNorm = np.zeros(imgEqualized.shape)
        imgNormalized = cv2.normalize(imgEqualized, imgNorm, 0, 255, cv2.NORM_MINMAX)
        kernel_blur = np.ones((3, 3), np.float32) / 9
        imgBlur = cv2.filter2D(imgNormalized, -1, kernel_blur)
        imgFinal = imgBlur
        
        # Converter de volta para bytes (JPEG)
        # cv2.imencode retorna (sucesso, array de bytes)
        success, encoded_img = cv2.imencode('.jpg', imgFinal)
        if not success:
            raise ValueError("Erro ao codificar a imagem pré-processada.")
        
        return encoded_img.tobytes()
    
    except Exception as e:
        raise Exception(f"Erro ao pré-processar imagem: {str(e)}")

def process_extracted_text_with_gemini(extracted_text):
    comprehensible_text = extract_comprehensible_text(extracted_text)
    prompt = format_prompt(comprehensible_text)
    response = get_validation_parameters(prompt)
    return response, comprehensible_text

def analyze_image_with_gemini(image_data, mime_type='image/jpeg'):
    """
    Envia a imagem diretamente para o Gemini e recebe análise de meme.
    
    Args:
        image_data: bytes da imagem
        mime_type: tipo MIME da imagem (image/jpeg ou image/png)
        
    Returns:
        response: resposta da API do Gemini
    """
    # Converter imagem para base64
    image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    headers = {
        'Content-Type': 'application/json',
        'X-goog-api-key': GOOG_API_KEY
    }
    
    prompt = """Analise APENAS os elementos visuais desta imagem (sem considerar textos ou frases). Foque na composição visual, estilo, elementos gráficos, cores, formas e layout.

Instruções:
1. Analise os elementos visuais presentes na imagem (imagens, gráficos, composição, estilo visual, cores, formas, layout)
2. Para cada elemento visual identificado, analise:
   - A probabilidade de ser meme (valor de 0 a 100)
   - Um detalhamento explicando a análise baseada APENAS em elementos visuais

3. Além disso, faça uma análise geral considerando TODOS os elementos visuais em conjunto, avaliando a imagem completa como um todo, baseando-se APENAS em aspectos visuais (composição, estilo de meme, elementos gráficos, contexto visual).

4. Retorne APENAS um JSON válido no seguinte formato (sem markdown, sem explicações adicionais):

{
  "elemento_visual_1": {
    "probabilidade_de_ser_meme": <número de 0 a 100>,
    "detalhamento": "<explicação detalhada da análise baseada APENAS em elementos visuais>"
  },
  "elemento_visual_2": {
    "probabilidade_de_ser_meme": <número de 0 a 100>,
    "detalhamento": "<explicação detalhada da análise baseada APENAS em elementos visuais>"
  },
  "analise_geral": {
    "probabilidade_de_ser_meme": <número de 0 a 100, considerando todos os elementos visuais em conjunto>,
    "detalhamento": "<explicação detalhada da análise geral da imagem completa, considerando APENAS aspectos visuais como composição, estilo típico de memes, elementos gráficos, cores, formas, layout, contexto visual, etc. NÃO mencione textos ou frases.>"
  }
}

IMPORTANTE: 
- Analise APENAS elementos visuais (imagens, gráficos, composição, estilo, cores, formas, layout)
- NÃO analise ou mencione textos ou frases
- Retorne APENAS o JSON, sem markdown, sem código, sem explicações. Apenas o JSON puro."""
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": image_base64
                        }
                    },
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=payload)
    return response

@app.route('/generate', methods=['POST'])
def generate_content():

    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Campo "text" é obrigatório no corpo da requisição'}), 400
        
        user_text = data['text']
        
        prompt = format_prompt(user_text)
        
        response = get_validation_parameters(prompt)
        
        if response.status_code == 200:
            return jsonify(response.json()), 200
        else:
            return jsonify({
                'error': 'Erro na requisição para o Google Gemini API',
                'status_code': response.status_code,
                'message': response.text
            }), response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Rota unificada que executa todo o fluxo:
    1. Extração de texto da imagem (OCR)
    2. Processamento e limpeza do texto com Gemini
    3. Validação e análise de credibilidade com Gemini (baseado no texto)
    4. Análise direta da imagem com Gemini (análise visual)
    
    Aceita parâmetro opcional 'detailed=true' para incluir confiança do OCR
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem foi enviada. Use a chave "image" no form-data.'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'Nenhum arquivo foi selecionado.'}), 400
        
        image_data = image_file.read()
        
        if not image_data:
            return jsonify({'error': 'Arquivo de imagem vazio.'}), 400
        
        # Determinar o tipo MIME da imagem
        filename = image_file.filename.lower()
        if filename.endswith('.png'):
            mime_type = 'image/png'
        else:
            mime_type = 'image/jpeg'
        
        detailed = request.form.get('detailed', 'false').lower() == 'true'
        
        # 1. Extração de texto (OCR)
        if detailed:
            result = extract_text_with_confidence(image_data)
            extracted_text = result['text']
            ocr_confidence = result['confidence']
        else:
            extracted_text = extract_text_from_image(image_data)
            ocr_confidence = None
        
        # 2. Análise baseada no texto extraído
        text_response, comprehensible_text = process_extracted_text_with_gemini(extracted_text)
        
        # 3. Pré-processar a imagem antes de enviar ao Gemini
        preprocessed_image_data = preprocess_image(image_data)
        
        # 4. Análise direta da imagem (pré-processada)
        image_response = analyze_image_with_gemini(preprocessed_image_data, mime_type)
        
        # Processar resposta do texto
        text_analysis = None
        if text_response.status_code == 200:
            gemini_result = text_response.json()
            gemini_text = ""
            if 'candidates' in gemini_result and len(gemini_result['candidates']) > 0:
                if 'content' in gemini_result['candidates'][0]:
                    parts = gemini_result['candidates'][0]['content'].get('parts', [])
                    if parts and 'text' in parts[0]:
                        gemini_text = parts[0]['text']
            
            text_analysis = extract_json_from_response(gemini_text)
        
        # Processar resposta da imagem
        image_analysis = None
        if image_response.status_code == 200:
            gemini_result = image_response.json()
            gemini_text = ""
            if 'candidates' in gemini_result and len(gemini_result['candidates']) > 0:
                if 'content' in gemini_result['candidates'][0]:
                    parts = gemini_result['candidates'][0]['content'].get('parts', [])
                    if parts and 'text' in parts[0]:
                        gemini_text = parts[0]['text']
            
            image_analysis = extract_json_from_response(gemini_text)
        
        # Calcular médias e resultado final
        text_avg = calculate_average_probability(text_analysis) if text_analysis else None
        image_avg = calculate_average_probability(image_analysis) if image_analysis else None
        
        # Aplicar peso de 1.25 na média da análise de imagem
        if image_avg is not None:
            image_avg_weighted = image_avg * 1.25
            # Limitar a 100 (caso ultrapasse)
            image_avg_weighted = min(image_avg_weighted, 100)
        else:
            image_avg_weighted = None
        
        # Calcular média final: (média_texto + média_imagem_ponderada) / 2
        final_average = None
        if text_avg is not None and image_avg_weighted is not None:
            final_average = (text_avg + image_avg_weighted) / 2
        elif text_avg is not None:
            final_average = text_avg
        elif image_avg_weighted is not None:
            final_average = image_avg_weighted
        
        # Montar resposta
        response_data = {
            'success': True,
            'extracted_text': extracted_text,
            'comprehensible_text': comprehensible_text,
            'text_analysis': text_analysis if text_analysis else None,
            'image_analysis': image_analysis if image_analysis else None,
            'averages': {
                'text_average': round(text_avg, 2) if text_avg is not None else None,
                'image_average': round(image_avg, 2) if image_avg is not None else None,
                'image_average_weighted': round(image_avg_weighted, 2) if image_avg_weighted is not None else None,
                'final_average': round(final_average, 2) if final_average is not None else None
            }
        }
        
        if detailed:
            response_data['ocr_confidence'] = ocr_confidence
        
        # Verificar se houve erros
        if text_response.status_code != 200 or image_response.status_code != 200:
            response_data['success'] = False
            response_data['errors'] = {}
            if text_response.status_code != 200:
                response_data['errors']['text_analysis'] = {
                    'status_code': text_response.status_code,
                    'message': text_response.text
                }
            if image_response.status_code != 200:
                response_data['errors']['image_analysis'] = {
                    'status_code': image_response.status_code,
                    'message': image_response.text
                }
            
            status_code = max(text_response.status_code, image_response.status_code)
            return jsonify(response_data), status_code
        
        return jsonify(response_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():

    return jsonify({'status': 'ok'}), 200

def cli_mode():

    print("=== Modo CLI - Google Gemini API ===")
    print("Digite 'sair' ou 'exit' para encerrar\n")
    
    while True:
        try:
            user_input = input("Você: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("\nEncerrando...")
                break
            
            # Formata o prompt com o texto padrão
            prompt = format_prompt(user_input)
            
            print("\nProcessando...")
            response = get_validation_parameters(prompt)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and len(result['candidates']) > 0:
                    if 'content' in result['candidates'][0]:
                        parts = result['candidates'][0]['content'].get('parts', [])
                        if parts and 'text' in parts[0]:
                            print(f"\nGemini: {parts[0]['text']}\n")
                        else:
                            print(f"\nResposta: {result}\n")
                    else:
                        print(f"\nResposta: {result}\n")
                else:
                    print(f"\nResposta: {result}\n")
            else:
                print(f"\nErro: {response.status_code}")
                print(f"Mensagem: {response.text}\n")
                
        except KeyboardInterrupt:
            print("\n\nEncerrando...")
            break
        except Exception as e:
            print(f"\nErro: {str(e)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Servidor Flask para Google Gemini API')
    parser.add_argument('--cli', action='store_true', help='Executa em modo CLI para entrada manual')
    args = parser.parse_args()
    
    if args.cli:
        cli_mode()
    else:
        app.run(host='0.0.0.0', port=int(SERVER_PORT) | 5000)

