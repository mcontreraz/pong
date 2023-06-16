import pygame
from pygame.locals import *
import sys
import numpy as np
import tensorflow as tf

# Inicializar Pygame
pygame.init()

# Configuración de la pantalla y del juego
WIDTH, HEIGHT = 640, 480
BG_COLOR = (0, 0, 0)  # Color de fondo
FPS = 60  # Fotogramas por segundo

# Configuración de la raqueta y la pelota
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 60
BALL_RADIUS = 5
BALL_SPEED_X = 5  # Velocidad de la pelota en el eje X
BALL_SPEED_Y = 5  # Velocidad de la pelota en el eje Y

# Configuración de la red neuronal
INPUT_SIZE = 1000 # Tamaño del vector de entrada
HIDDEN_SIZE = 1000  # Tamaño de la capa oculta
OUTPUT_SIZE = 2000 # Tamaño de la salida (movimiento de la paleta)
LEARNING_RATE = 0.01  # Tasa de aprendizaje

# Crear la ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

# Definir las raquetas y la pelota
paddle1 = pygame.Rect(0, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
paddle2 = pygame.Rect(WIDTH - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT)
ball = pygame.Rect(WIDTH // 2 - BALL_RADIUS // 2, HEIGHT // 2 - BALL_RADIUS // 2, BALL_RADIUS, BALL_RADIUS)

# Definir la red neuronal
input_layer = tf.keras.layers.Input(shape=(INPUT_SIZE,))
hidden_layer = tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu')(input_layer)
output_layer = tf.keras.layers.Dense(OUTPUT_SIZE, activation='sigmoid')(hidden_layer)
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=optimizer, loss='binary_crossentropy')

# Cargar el modelo pre-entrenado
try:
    model.load_weights('pong_model.h5')
    print("Modelo cargado exitosamente.")
except:
    print("No se encontró el modelo pre-entrenado.")

# Función para obtener el estado actual del juego
def get_game_state():
    ball_x = ball.x / WIDTH
    ball_y = ball.y / HEIGHT
    paddle1_y = paddle1.y / HEIGHT
    return np.array([ball_x, ball_y, paddle1_y, paddle2.y])

clock = pygame.time.Clock()

score = 0

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, (255, 255, 255), paddle1)
    pygame.draw.rect(screen, (255, 255, 255), paddle2)
    pygame.draw.ellipse(screen, (255, 255, 255), ball)
    pygame.display.flip()
    clock.tick(FPS)

    # Movimiento de la paleta del jugador 1 (controlado por el usuario)
    keys = pygame.key.get_pressed()
    if keys[K_w]:
        paddle1.y -= 5
    elif keys[K_s]:
        paddle1.y += 5

    # Movimiento de la paleta del jugador 2 (controlado por la red neuronal)
    game_state = get_game_state()
    predicted_movement = model.predict(np.expand_dims(game_state, axis=0))[0][0]
    if predicted_movement < 0.5:
        paddle2.y -= 5
    elif predicted_movement > 0.5:
        paddle2.y += 5

    # Controlar los límites de las paletas
    paddle1.y = max(min(paddle1.y, HEIGHT - PADDLE_HEIGHT), 0)
    paddle2.y = max(min(paddle2.y, HEIGHT - PADDLE_HEIGHT), 0)


    # Controlar los límites de las paletas
    if paddle1.top < 0:
        paddle1.top = 0
    if paddle1.bottom > HEIGHT:
        paddle1.bottom = HEIGHT
    if paddle2.top < 0:
        paddle2.top = 0
    if paddle2.bottom > HEIGHT:
        paddle2.bottom = HEIGHT

    # Mover la pelota (código omitido para mayor claridad)
    ball.x += BALL_SPEED_X
    ball.y += BALL_SPEED_Y

    # Controlar las colisiones de la pelota con las paredes
    if ball.top < 0 or ball.bottom > HEIGHT:
        BALL_SPEED_Y *= -1
    if ball.left < 0:
        BALL_SPEED_X *= -1
        score += 1
    if ball.right > WIDTH:
        BALL_SPEED_X *= -1
        score = 0

    # Controlar las colisiones de la pelota con las raquetas
    if ball.colliderect(paddle1) or ball.colliderect(paddle2):
        BALL_SPEED_X *= -1

    # Dibujar la pantalla (código omitido para mayor claridad)

    # Mostrar el puntaje
    font = pygame.font.Font(None, 36)
    text = font.render("Score: " + str(score), True, (255, 255, 255))
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(FPS)
