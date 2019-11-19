import numpy as np
import pygame

n_x = 7              # 7 input nodes
n_h = 9              # 9 nodes in hidden layer 1
n_h2 = 15            # 15 nodes in hidden layer 2
n_y = 3              # 3 output nodes
w1_shape = [n_h, n_x]    # 7 input, 9 hidden
w2_shape = [n_h2, n_h]   # 9 hidden input, 15 hidden
w3_shape =[n_y, n_h2]    # 15 hidden input, 3 output


def get_weights_from_encoded(individual):
    w1 = individual[0:w1_shape[0] * w1_shape[1]]
    w2 = individual[w1_shape[0] * w1_shape[1]:w2_shape[0] * w2_shape[1] + w1_shape[0] * w1_shape[1]]
    w3 = individual[w2_shape[0] * w2_shape[1] + w1_shape[0] * w1_shape[1]:]

    return (w1.reshape(w1_shape[0], w1_shape[1])), w2.reshape(w2_shape[0], w2_shape[1]), w3.reshape(w3_shape[0], w3_shape[1])


def soft_max(z):
    return np.exp(z.T) / np.sum(np.exp(z.T), axis=1).reshape(-1,1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation(x, individual, screen=None):
    w1, w2, w3 = get_weights_from_encoded(individual)

    z1 = np.matmul(w1, x.T)
    a1 = np.tanh(z1)

    z2 = np.matmul(w2, a1)
    a2 = np.tanh(z2)

    z3 = np.matmul(w3, a2)
    a3 = soft_max(z3)

    if screen is not None:
        draw_network(screen, x, a1, a2, a3, w1, w2, w3)
    return a3


''' whoever looks at the following code, i'm sorry '''
def draw_network(screen, x, a1, a2, a3, w1, w2, w3):
    layer_1_x_offset = 350
    layer_2_x_offset = 400
    layer_3_x_offset = 450
    layer_4_x_offset = 500
    for i in range(len(a1)):
        for j in range(len(x[0])):
            if w1[i][j] < 0:
                if w1[i][j] < -1:
                    pygame.draw.line(screen, (0, 0, 255),
                                     (layer_1_x_offset, 40 + (j * 20)), (layer_2_x_offset, 22 + (i * 20)))
                else:
                    pygame.draw.line(screen, (255 + w1[i][j] * 255, 255 + w1[i][j] * 255, 255),
                                     (layer_1_x_offset, 40 + (j * 20)), (layer_2_x_offset, 22 + (i * 20)))
            else:
                if w1[i][j] > 1:
                    pygame.draw.line(screen, (255, 0, 0),
                                     (layer_1_x_offset, 40 + (j * 20)), (layer_2_x_offset, 22 + (i * 20)))
                else:
                    pygame.draw.line(screen, (255, 255 - w1[i][j] * 255, 255 - w1[i][j] * 255),
                                     (layer_1_x_offset, 40 + (j * 20)), (layer_2_x_offset, 22 + (i * 20)))
    for i in range(len(a2)):
        for j in range(len(a1)):
            if w2[i][j] < 0:
                if w2[i][j] < -1:
                    pygame.draw.line(screen, (0, 0, 255),
                                     (layer_2_x_offset, 22 + (j * 20)), (layer_3_x_offset, 11 + (i * 13)))
                else:
                    pygame.draw.line(screen, (255 + w2[i][j] * 255, 255 + w2[i][j] * 255, 255),
                                     (layer_2_x_offset, 22 + (j * 20)), (layer_3_x_offset, 11 + (i * 13)))
            else:
                if w2[i][j] > 1:
                    pygame.draw.line(screen, (255, 0, 0),
                                     (layer_2_x_offset, 22 + (j * 20)), (layer_3_x_offset, 11 + (i * 13)))
                else:
                    pygame.draw.line(screen, (255, 255 - w2[i][j] * 255, 255 - w2[i][j] * 255),
                                     (layer_2_x_offset, 22 + (j * 20)), (layer_3_x_offset, 11 + (i * 13)))
    for i in range(len(a3[0])):
        for j in range(len(a2)):
            if w3[i][j] < 0:
                if w3[i][j] < -1:
                    pygame.draw.line(screen, (0, 0, 255),
                                     (layer_3_x_offset, 11 + (j * 13)), (layer_4_x_offset, 75 + (i * 25)))
                else:
                    pygame.draw.line(screen, (255 + w3[i][j] * 255, 255 + w3[i][j] * 255, 255),
                                     (layer_3_x_offset, 11 + (j * 13)), (layer_4_x_offset, 75 + (i * 25)))
            else:
                if w3[i][j] > 1:
                    pygame.draw.line(screen, (255, 0, 0),
                                     (layer_3_x_offset, 11 + (j * 13)), (layer_4_x_offset, 75 + (i * 25)))
                else:
                    pygame.draw.line(screen, (255, 255 - w3[i][j] * 255, 255 - w3[i][j] * 255),
                                     (layer_3_x_offset, 11 + (j * 13)), (layer_4_x_offset, 75 + (i * 25)))

    labels = ['left_blocked', 'front_blocked', 'right_blocked',
              'food vector(x)', 'direction vector(x)',
              'food vector(y)', 'direction vector(y)']
    for i in range(len(x[0])):
        if x[0][i] < 0:
            pygame.draw.circle(screen, (255 + x[0][i] * 255, 255 + x[0][i] * 255, 255), (layer_1_x_offset, 40 + (i * 20)), 5)
        else:
            pygame.draw.circle(screen, (255, 255 - (x[0][i] * 255), 255 - (x[0][i] * 255)), (layer_1_x_offset, 40 + (i * 20)), 5)

        font = pygame.font.SysFont('monospace', 10)
        text_surface = font.render(labels[i], True, (255,255,255))
        screen.blit(text_surface, (layer_1_x_offset - 125, 33 + (i * 20)))

    for i in range(len(a1)):
        if a1[i] < 0:
            pygame.draw.circle(screen, (255 + a1[i] * 255, 255 + a1[i] * 255, 255), (layer_2_x_offset, 22 + (i * 20)), 5)
        else:
            pygame.draw.circle(screen, (255, 255 - a1[i] * 255, 255 - a1[i] * 255), (layer_2_x_offset, 22 + (i * 20)), 5)

    for i in range(len(a2)):
        if a2[i] < 0:
            pygame.draw.circle(screen, (255 + a2[i] * 255, 255 + a2[i] * 255, 255), (layer_3_x_offset, 11 + (i * 13)), 5)
        else:
            pygame.draw.circle(screen, (255, 255 - a2[i] * 255, 255 - a2[i] * 255), (layer_3_x_offset, 11 + (i * 13)), 5)

    labels = ['left', 'straight', 'right']
    for i in range(len(a3[0])):
        if a3[0][i] < 0:
            pygame.draw.circle(screen, (255 + a3[0][i] * 255, 255 + a3[0][i] * 255, 255), (layer_4_x_offset, 75 + (i * 25)), 5)
        else:
            pygame.draw.circle(screen, (255, 255 - a3[0][i] * 255, 255 - a3[0][i] * 255), (layer_4_x_offset, 75 + (i * 25)), 5)

        font = pygame.font.SysFont('monospace', 10)
        text_surface = font.render(labels[i], True, (255, 255, 255))
        screen.blit(text_surface, (layer_4_x_offset + 8, 70 + (i * 25)))

    font = pygame.font.SysFont('monospace', 15)
    text_surface = font.render('blue: -ve value', True, (255,255,255))
    screen.blit(text_surface, (layer_1_x_offset - 125, 180))
    text_surface = font.render('red:  +ve value', True, (255,255,255))
    screen.blit(text_surface, (layer_1_x_offset - 125, 190))
    text_surface = font.render('white:  0 value', True, (255,255,255))
    screen.blit(text_surface, (layer_1_x_offset - 125, 200))
    pygame.display.update()

