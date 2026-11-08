import cv2 as cv
import numpy as np
import os
from moviepy import VideoFileClip
from pathlib import Path

#A ordem não importa, desde que seja mantida
video_esq_path = Path(r"data\ip13_v2.mov")
video_dir_path = Path(r"data\ip17_v2.mov")

pastas = ['calibracao_esq', 'calibracao_dir', 'sinc_esq', 'sinc_dir']
for p in pastas:
    caminho = os.path.join("data", p)
    os.makedirs(caminho, exist_ok=True)

def extrair_janela(cap, start_sec, end_sec, pasta_saida, prefixo):
    tempo_atual = start_sec
    count = 0
    while tempo_atual <= end_sec:
        # Pula diretamente para o milissegundo exato
        cap.set(cv.CAP_PROP_POS_MSEC, tempo_atual * 1000)
        ret, frame = cap.read()
        if ret:
            nome = f"{prefixo}_{count:03d}.png"
            cv.imwrite(os.path.join(pasta_saida, nome), frame)
            count += 1
        tempo_atual += intervalo_extracao

#1. Sincronização por áudio.
print("Carregando trilhas de áudio")
clip_esq = VideoFileClip(video_esq_path)
clip_dir = VideoFileClip(video_dir_path)

audio_esq = clip_esq.audio.to_soundarray(fps=44100)
audio_dir = clip_dir.audio.to_soundarray(fps=44100)

limite_segundos = 40 
limite_indices = limite_segundos * 44100

pico_idx_esq = np.argmax(np.abs(audio_esq[:limite_indices, 0])) 
pico_idx_dir = np.argmax(np.abs(audio_dir[:limite_indices, 0]))

#Converte o índice para segundos
tempo_palma_esq = pico_idx_esq / 44100.0
tempo_palma_dir = pico_idx_dir / 44100.0

print(f"Palma detectada - Esq: {tempo_palma_esq:.3f}s | Dir: {tempo_palma_dir:.3f}s")

# Libera os arquivos de áudio da memória
clip_esq.close()
clip_dir.close()

#2. Definição das janelas de extração -> deve ser feito manualmente e alterado para cada vídeo

#Definição das janelas para calibração individual -> o intervalo deve ser igual para ambos os vídeos
tempos_calib_dir = (33, 40) #tempo absoluto da própria câmera 
tempos_calib_esq = (27, 34) #tempo absoluto da própria câmera 

tempos_sinc = (59, 83) #Em relação ao vídeo ESQUERDO

#A sincronização por áudio não é tão precisa e os melhores frames para calibração terão que ser escolhidos manualmente
#Faz sentido extrair muitos frames e filtrar manualmente os frames bons -> RMSE MENOR
intervalo_extracao = 0.7

offset = tempo_palma_dir - tempo_palma_esq

cap_esq = cv.VideoCapture(video_esq_path)
cap_dir = cv.VideoCapture(video_dir_path)

print("Extraindo imagens de calibração individual - Esquerda")
extrair_janela(cap_esq, tempos_calib_esq[0], tempos_calib_esq[1], os.path.join('data', 'calibracao_esq'), 'esq')

print("Extraindo imagens de calibração individual - Direita")
extrair_janela(cap_dir, tempos_calib_dir[0], tempos_calib_dir[1], os.path.join('data', 'calibracao_dir'), 'dir')

print("Extraindo pares sincronizados")
tempo_atual = tempos_sinc[0]
count = 0
while tempo_atual <= tempos_sinc[1]:
    # Tempo na câmera esquerda -> Referência
    t_esq = tempo_atual
    # Tempo correspondente na câmera direita aplicando o offset matemático
    t_dir = tempo_atual + offset
    
    cap_esq.set(cv.CAP_PROP_POS_MSEC, t_esq * 1000)
    cap_dir.set(cv.CAP_PROP_POS_MSEC, t_dir * 1000)
    
    ret_esq, frame_esq = cap_esq.read()
    ret_dir, frame_dir = cap_dir.read()
    
    if ret_esq and ret_dir:
        nome = f"sinc_{count:04d}.png"
        cv.imwrite(os.path.join('data', 'sinc_esq', nome), frame_esq)
        cv.imwrite(os.path.join('data', 'sinc_dir', nome), frame_dir)
        count += 1
        
    tempo_atual += intervalo_extracao

cap_esq.release()
cap_dir.release()
print(f"Processo finalizado. {count} pares extraídos.")