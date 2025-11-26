INSTALAÇÃO----------------
1-> Instalar as dependências do projeto:
pip install requirements.txt

2-> Pegar a chave da API para acessar o seu modelo online:
aistudio.google.com
    2.2-> criar chave de api
    2.3-> substituir em ./.env GOOG_API_KEY pela chave que acessa o modelo
    2.4 (opcional) -> configurar a porta do servidor localhost, caso outra
    aplicação já a esteja utilizando
--------------------------

INICIAR O SERVIDOR--------
Modo 1: Inputar diretamente o prompt via linha de comando:
cmd
    python3 server.py --cli

Nesse modo, a prompt é um input direto através da mesma cli utilizada
para iniciar o servidor

Modo 2: Iniciar o servidor, esperando a requisição
cmd
    python3 server.py
    
Nesse modo, a requisição terá que ser feita na rota /generate. A
requisição para essa rota terá o seguinte corpo:
/generate
   curl -X POST http://localhost:5000/generate \
     -H "Content-Type: application/json" \
     -d '{"text": "Explique como a IA funciona em poucas palavras"}'

O script vai pegar o corpo da requisição (campo text) e adequar ao seguinte modelo:
curl "https://generativelanguage.googleapis.com/v1beta/models/<modelo>" \
  -H 'Content-Type: application/json' \
  -H 'X-goog-api-key: <chaveapidogoogle>' \
  -X POST \
  -d '{
    "contents": [
      {
        "parts": [
          {
            "text": "<promptinputada>"
          }
        ]
      }
    ]
  }'

EXTRAÇÃO DE TEXTO DE IMAGENS (OCR)--------
3-> Instalar o Tesseract OCR no sistema:
    Linux: sudo apt-get install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng
    macOS: brew install tesseract tesseract-lang
    Windows: Baixe do GitHub (https://github.com/UB-Mannheim/tesseract/wiki)

4-> Endpoint /extract-text - Extração simples de texto:
    curl -X POST http://localhost:5000/extract-text \
      -F "image=@/caminho/para/sua/imagem.png"

    Exemplo com arquivo local:
    curl -X POST http://localhost:5000/extract-text \
      -F "image=@./data/imagem.jpg"

5-> Endpoint /extract-text-detailed - Extração com confiança:
    curl -X POST http://localhost:5000/extract-text-detailed \
      -F "image=@/caminho/para/sua/imagem.png"

    Exemplo com arquivo local:
    curl -X POST http://localhost:5000/extract-text-detailed \
      -F "image=@./data/imagem.jpg"

    Resposta esperada:
    {
      "success": true,
      "text": "Texto extraído da imagem",
      "confidence": 95.5
    }

Nota: O campo "image" é obrigatório e deve conter o arquivo de imagem.
Formatos suportados: PNG, JPG, JPEG, GIF, BMP, TIFF, etc.

