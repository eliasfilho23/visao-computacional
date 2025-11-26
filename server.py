import os
import requests
import argparse
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
    prompt = f"""me diga pra mim a probabilidade das seguintes palavras serem um meme ou serem uma abordagem séria: {user_text}

Por favor, retorne uma métrica de credibilidade de 1 (muito falso) a 5 (muito verdadeiro) para avaliar a credibilidade do texto."""
    return prompt

def send_text_to_gemini(user_text):

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

@app.route('/generate', methods=['POST'])
def generate_content():

    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Campo "text" é obrigatório no corpo da requisição'}), 400
        
        user_text = data['text']
        
        prompt = format_prompt(user_text)
        
        response = send_text_to_gemini(prompt)
        
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

@app.route('/extract-text', methods=['POST'])
def extract_text():

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem foi enviada. Use a chave "image" no form-data.'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'Nenhum arquivo foi selecionado.'}), 400
        
        image_data = image_file.read()
        
        if not image_data:
            return jsonify({'error': 'Arquivo de imagem vazio.'}), 400
        
        extracted_text = extract_text_from_image(image_data)
        
        return jsonify({
            'success': True,
            'text': extracted_text
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract-text-detailed', methods=['POST'])
def extract_text_detailed():

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem foi enviada. Use a chave "image" no form-data.'}), 400
        
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'error': 'Nenhum arquivo foi selecionado.'}), 400
        
        image_data = image_file.read()
        
        if not image_data:
            return jsonify({'error': 'Arquivo de imagem vazio.'}), 400
        
        result = extract_text_with_confidence(image_data)
        
        return jsonify({
            'success': True,
            'text': result['text'],
            'confidence': result['confidence']
        }), 200
        
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
            response = send_text_to_gemini(prompt)
            
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

