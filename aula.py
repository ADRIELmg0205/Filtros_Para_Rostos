import cv2

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
face_scale_factor = 1.0  # Fator de escala inicial para o rosto

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame capturado para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Verificar teclas pressionadas para trocar o emoji ou ajustar o tamanho do rosto
    key = cv2.waitKey(1)
    if key in emojis:
        current_emoji = emojis[key]
    elif key == ord('+'):
        face_scale_factor += 0.1  # Aumenta o fator de escala do rosto
    elif key == ord('-'):
        if face_scale_factor > 0.1:  # Evita tamanho negativo
            face_scale_factor -= 0.1  # Diminui o fator de escala do rosto

    if current_emoji:
        emoji_img = cv2.imread(current_emoji, -1)

        # Aplicar o filtro de emoji nos rostos detectados
        for (x, y, w, h) in faces:
            w_with_scale = int(w * face_scale_factor)  # Aplica o fator de escala no tamanho do rosto
            h_with_scale = int(h * face_scale_factor)

            x_offset = int((w_with_scale - w) / 2)  # Calcula o deslocamento para manter o centro
            y_offset = int((h_with_scale - h) / 2)

            emoji_resized = cv2.resize(emoji_img, (w_with_scale, h_with_scale))

            # Aplicar o emoji na região do rosto
            for i in range(emoji_resized.shape[0]):
                for j in range(emoji_resized.shape[1]):
                    if emoji_resized[i, j][3] != 0 and y + i - y_offset < frame.shape[0] and x + j - x_offset < frame.shape[1]:
                        frame[y + i - y_offset, x + j - x_offset] = emoji_resized[i, j, :3]

    # Exibir o vídeo com o filtro de emoji
    cv2.imshow('Emoji Filter', frame)

    # Pressione 'Esc' para sair do loop
    if key == 27:
        break

# Encerrar a captura e fechar as janelas
cap.release()
cv2.destroyAllWindows()
