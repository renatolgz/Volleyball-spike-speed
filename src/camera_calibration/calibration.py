import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import os

"""
~95% desse código é derivado do tutorial do Temuge Batpurev
Toda a explicação pode ser encontrada no link: https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
"""

#Dimensões do tabuleiro -> sempre 1 a menos que a quantidade de quadrados (Meu tabuleiro é 10 x 6)
COLUMNS = 9
ROWS = 5
WORLD_SCALING = 0.035 #tamanho em metros do lado dos quadrados (35 mm)

#Flags obrigatórias para iPhones
FLAGS = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
CONV_SIZE = (11, 11)

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

def calibrate_camera(images_folder):
    """
    Calibração individual de cada uma das câmeras prévia a calibração conjunta
    """
    images_names = sorted(glob.glob(images_folder))
    images = []
    
    for imname in images_names:
        im = cv.imread(imname, 1)
        #Redimensionamento (os vídeos foram gravados em 4K)
        im_resized = cv.resize(im, (0,0), fx=0.5, fy=0.5)
        images.append(im_resized)
 
    objp = np.zeros((ROWS*COLUMNS, 3), np.float32)
    objp[:,:2] = np.mgrid[0:COLUMNS, 0:ROWS].T.reshape(-1,2)
    objp = WORLD_SCALING * objp
 
    width = images[0].shape[1]
    height = images[0].shape[0]
 
    imgpoints = [] 
    objpoints = [] 
 
    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (COLUMNS, ROWS), flags=FLAGS)
 
        if ret == True:
            corners = cv.cornerSubPix(gray, corners, CONV_SIZE, (-1, -1), CRITERIA)
            objpoints.append(objp)
            imgpoints.append(corners)
 
    #Calcula as matrizes intrínsecas da câmera
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print(f'RMSE da calibração individual ({images_folder}):', ret)
 
    return mtx, dist
 
#1. Calibração Individual -> Podem ser utilizadas as mesmas imagens da calibração com duas câmeras
#folder_esq = os.path.join('data', 'calibracao_esq','*')
#folder_dir = os.path.join('data', 'calibracao_dir','*')
folder_esq = os.path.join('data', 'sinc_esq', '*')
folder_dir = os.path.join('data', 'sinc_dir', '*')

mtx1, dist1 = calibrate_camera(images_folder = folder_esq) 
mtx2, dist2 = calibrate_camera(images_folder = folder_dir)

def stereo_calibrate(mtx1, dist1, mtx2, dist2, folder_esq, folder_dir, debug_xadrez=False, debug_epipolar=False):
    c1_names = sorted(glob.glob(folder_esq))
    c2_names = sorted(glob.glob(folder_dir))

    #Pareamento por numeração de arquivo
    pares_validos = []
    for im1_path in c1_names:
        nome_arq = os.path.basename(im1_path)
        im2_path = next((p for p in c2_names if os.path.basename(p) == nome_arq), None)
        
        if im2_path is not None:
            pares_validos.append((im1_path, im2_path))

    print(f"Pares válidos: {len(pares_validos)}")

    objp = np.zeros((ROWS*COLUMNS, 3), np.float32)
    objp[:,:2] = np.mgrid[0:COLUMNS, 0:ROWS].T.reshape(-1,2)
    objp = WORLD_SCALING * objp

    imgpoints_left = [] 
    imgpoints_right = []
    objpoints = [] 

    width, height = None, None
    pares_sucesso = 0

    for im1_path, im2_path in pares_validos:

        frame1 = cv.imread(im1_path, 1)
        frame2 = cv.imread(im2_path, 1)
        
        #resize
        frame1 = cv.resize(frame1, (0,0), fx=0.5, fy=0.5)
        frame2 = cv.resize(frame2, (0,0), fx=0.5, fy=0.5)

        if width is None:
            width = frame1.shape[1]
            height = frame1.shape[0]

        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (COLUMNS, ROWS), flags=FLAGS)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (COLUMNS, ROWS), flags=FLAGS)

        if c_ret1 and c_ret2:
            corners1 = cv.cornerSubPix(gray1, corners1, CONV_SIZE, (-1, -1), CRITERIA)
            corners2 = cv.cornerSubPix(gray2, corners2, CONV_SIZE, (-1, -1), CRITERIA)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            pares_sucesso += 1

            #Debug para verificação se os tabuleiros foram identificados no mesmo sentido.
            #Frames onde a ordem das cores de exibição não for igual devem ser descartados    
            if debug_xadrez: 
                debug_esq = frame1.copy()
                debug_dir = frame2.copy()
                
                # Desenha a ordem dos pontos detectados
                cv.drawChessboardCorners(debug_esq, (COLUMNS, ROWS), corners1, c_ret1)
                cv.drawChessboardCorners(debug_dir, (COLUMNS, ROWS), corners2, c_ret2)
                
                #Redimensionamento das imagens de debug
                tela_esq = cv.resize(debug_esq, (0, 0), fx=0.35, fy=0.35)
                tela_dir = cv.resize(debug_dir, (0, 0), fx=0.35, fy=0.35)
                
                cv.imshow('Ordem Esquerda', tela_esq)
                cv.imshow('Ordem Direita', tela_dir)
                print(f"Exibindo par {pares_sucesso}. Pressione qualquer tecla na janela do vídeo para avançar...")
                cv.waitKey(0) 

        else:
            print(f"Descartado: Xadrez não detectado em ambas as lentes no frame {os.path.basename(im1_path)}")

    print(f"Calibrando com {pares_sucesso} pares espaciais perfeitos.")

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right, mtx1, dist1, mtx2, dist2, 
        (width, height), criteria = CRITERIA, flags = stereocalibration_flags
    )

    print('RMSE Estéreo:', ret)

    if debug_epipolar:
        #Permite visualizar a projeção e analisar a qualidade de cada frame de calibração
        #Na primeira calibração é necessário analisar os frames e excluir aqueles onde as linhas não estão perfeitamente alinhadas
        passo = 10 
    
        for im1_path, im2_path in pares_validos:
            img1_bruta = cv.imread(im1_path)
            img2_bruta = cv.imread(im2_path)
            
            #Redimensionamento
            img1 = cv.resize(img1_bruta, (0, 0), fx=0.5, fy=0.5)
            img2 = cv.resize(img2_bruta, (0, 0), fx=0.5, fy=0.5)
    
            ret1, corners1 = cv.findChessboardCorners(cv.cvtColor(img1, cv.COLOR_BGR2GRAY), (COLUMNS, ROWS))
            ret2, corners2 = cv.findChessboardCorners(cv.cvtColor(img2, cv.COLOR_BGR2GRAY), (COLUMNS, ROWS))
    
            if ret1 and ret2:
                linhas = cv.computeCorrespondEpilines(corners1.reshape(-1, 1, 2), 1, F)
                linhas = linhas.reshape(-1, 3)
    
                linhas_amostra = linhas[::passo]
                corners2_amostra = corners2[::passo]
    
                for r, pt2 in zip(linhas_amostra, corners2_amostra):
                    cor = tuple(np.random.randint(50, 255, 3).tolist())
                    x0, y0 = map(int, [0, -r[2]/r[1]])
                    x1, y1 = map(int, [img2.shape[1], -(r[2]+r[0]*img2.shape[1])/r[1]])
                    
                    cv.line(img2, (x0, y0), (x1, y1), cor, 2)
                    cv.circle(img2, tuple(np.int32(pt2[0])), 6, cor, -1)
    
                cv.putText(img2, os.path.basename(im2_path), (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imshow('Teste Epipolar Sequencial', img2)
                
                if cv.waitKey(0) & 0xFF == ord('q'):
                    break

    return R, T

R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, folder_esq, folder_dir, debug_xadrez=False, debug_epipolar=False)
#debugs = True para debug

RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1 
 
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtx2 @ RT2 

#Salva as matrizes de projeção
np.savez(os.path.join('data', 'matrizes_projetivas.npz'), P1=P1, P2=P2)
print("Matrizes salvas com sucesso em 'matrizes_projetivas.npz'")