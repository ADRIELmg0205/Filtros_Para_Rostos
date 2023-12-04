import cv2
import numpy as np

# Carregar o classificador para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

emojis = {
    ord('1'): 'venv/imagens/glasses.png',  # Substitua pelos caminhos das suas imagens de emojis
    ord('2'): 'venv/imagens/emoji.png',
    ord('3'): 'venv/imagens/nerd.png',
    ord('4'): 'venv/imagens/mascara_jason.png',
    ord('5'): 'venv/imagens/juliet.png',
    ord('6'): 'venv/imagens/armacao.png',
    ord('7'): 'venv/imagens/will_smith.png',
}

current_emoji = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame capturado para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Verificar teclas pressionadas para trocar o emoji
    key = cv2.waitKey(1)
    if key in emojis:
        current_emoji = emojis[key]

    if current_emoji:
        emoji_img = cv2.imread(current_emoji, -1)
        emoji_img = cv2.resize(emoji_img, (50, 50))

        # Aplicar o filtro de emoji nos rostos detectados
        for (x, y, w, h) in faces:
            emoji_resized = cv2.resize(emoji_img, (w, h))
            for i in range(emoji_resized.shape[0]):
                for j in range(emoji_resized.shape[1]):
                    if emoji_resized[i, j][3] != 0:
                        frame[y + i, x + j] = emoji_resized[i, j, :3]

    # Exibir o vídeo com o filtro de emoji
    cv2.imshow('Emoji Filter', frame)

    # Pressione 'q' para sair do loop
    if key & 0xFF == ord('q'):
        break

# Encerrar a captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()
