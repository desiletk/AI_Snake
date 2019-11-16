import numpy as np
import pygame

n_x = 7              # 7 input nodes
n_h = 9              # 9 nodes in hidden layer 1
n_h2 = 15            # 15 nodes in hidden layer 2
n_y = 3              # 3 output nodes
w1_shape = [9, 7]    # 7 input, 9 hidden
w2_shape = [15, 9]   # 9 hidden input, 15 hidden
w3_shape =[3, 15]    # 15 hidden input, 3 output


def get_weights_from_encoded(individual):
    w1 = individual[0:w1_shape[0] * w1_shape[1]]
    w2 = individual[w1_shape[0] * w1_shape[1]:w2_shape[0] * w2_shape[1] + w1_shape[0] * w1_shape[1]]
    w3 = individual[w2_shape[0] * w2_shape[1] + w1_shape[0] * w1_shape[1]:]

    return (w1.reshape(w1_shape[0], w1_shape[1])), w2.reshape(w2_shape[0], w2_shape[1]), w3.reshape(w3_shape[0], w3_shape[1])


def softmax(z):
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
    a3 = softmax(z3)

    if screen is not None:
        draw_network(screen, x, a1, a2, a3, w1, w2, w3)
    return a3


def draw_network(screen, x, a1, a2, a3, w1, w2, w3):
    layer_1_x_offset = 225
    layer_2_x_offset = 275
    layer_3_x_offset = 325
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
                    pygame.draw.line(screen, (255, 0, 0), (275, 22 + (j * 20)), (325, 11 + (i * 13)))
                else:
                    pygame.draw.line(screen, (255, 255 - w2[i][j] * 255, 255 - w2[i][j] * 255), (275, 22 + (j * 20)),
                                     (325, 11 + (i * 13)))
    for i in range(len(a3[0])):
        for j in range(len(a2)):
            if w3[i][j] < 0:
                if w3[i][j] < -1:
                    pygame.draw.line(screen, (0, 0, 255), (325, 11 + (j * 13)), (375, 75 + (i * 25)))
                else:
                    pygame.draw.line(screen, (255 + w3[i][j] * 255, 255 + w3[i][j] * 255, 255), (325, 11 + (j * 13)),
                                     (375, 75 + (i * 25)))
            else:
                if w3[i][j] > 1:
                    pygame.draw.line(screen, (255, 0, 0), (325, 11 + (j * 13)), (375, 75 + (i * 25)))
                else:
                    pygame.draw.line(screen, (255, 255 - w3[i][j] * 255, 255 - w3[i][j] * 255), (325, 11 + (j * 13)),
                                     (375, 75 + (i * 25)))

    for i in range(len(x[0])):
        if x[0][i] < 0:
            pygame.draw.circle(screen, (255 + x[0][i] * 255, 255 + x[0][i] * 255, 255), (225, 40 + (i * 20)), 5)
        else:
            pygame.draw.circle(screen, (255, 255 - (x[0][i] * 255), 255 - (x[0][i] * 255)), (225, 40 + (i * 20)), 5)

    for i in range(len(a1)):
        if a1[i] < 0:
            pygame.draw.circle(screen, (255 + a1[i] * 255, 255 + a1[i] * 255, 255), (275, 22 + (i * 20)), 5)
        else:
            pygame.draw.circle(screen, (255, 255 - a1[i] * 255, 255 - a1[i] * 255), (275, 22 + (i * 20)), 5)

    for i in range(len(a2)):
        if a2[i] < 0:
            pygame.draw.circle(screen, (255 + a2[i] * 255, 255 + a2[i] * 255, 255), (325, 11 + (i * 13)), 5)
        else:
            pygame.draw.circle(screen, (255, 255 - a2[i] * 255, 255 - a2[i] * 255), (325, 11 + (i * 13)), 5)

    for i in range(len(a3[0])):
        if a3[0][i] < 0:
            pygame.draw.circle(screen, (255 + a3[0][i] * 255, 255 + a3[0][i] * 255, 255), (375, 75 + (i * 25)), 5)
        else:
            pygame.draw.circle(screen, (255, 255 - a3[0][i] * 255, 255 - a3[0][i] * 255), (375, 75 + (i * 25)), 5)

        font = pygame.font.SysFont('monospace', 10)
        if i == 0:
            text_surface = font.render('left', True, (255, 255, 255))
        elif i == 1:
            text_surface = font.render('straight', True, (255, 255, 255))
        else:
            text_surface = font.render('right', True, (255, 255, 255))
        screen.blit(text_surface, (385, 75 + (i * 25)))

    pygame.display.update()

