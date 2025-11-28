#!/bin/bash
#./save_result.sh ./data/meme-2012.png

if [ -z "$1" ]; then
    echo "Uso: $0 <arquivo-imagem> [nome-resultado]"
    echo "Exemplo: $0 ./data/meme-2012.png meme-2012"
    exit 1
fi

IMAGE_FILE="$1"
RESULT_NAME="${2:-$(basename "$IMAGE_FILE" | sed 's/\.[^.]*$//')}"

mkdir -p results

echo "Processando imagem: $IMAGE_FILE"
echo "Salvando resultado em: results/${RESULT_NAME}-result.json"

curl -X POST "http://localhost:5000/analyze-image" \
    -F "image=@${IMAGE_FILE}" \
    -F "detailed=true" \
    | python3 -m json.tool > "results/${RESULT_NAME}-result.json"

if [ $? -eq 0 ]; then
    echo "✓ Resultado salvo com sucesso!"
    echo "Arquivo: results/${RESULT_NAME}-result.json"
else
    echo "✗ Erro ao processar imagem"
    exit 1
fi

