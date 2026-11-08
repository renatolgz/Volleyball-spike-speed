import torch
import cv2
import numpy as np
import json

from tracknet.infer_on_video import read_video, infer_model, remove_outliers, split_track, interpolation, write_track, infer_on_video
from tracknet.model import BallTrackerNet


def processar_camera(caminho_video, caminho_saida, modelo):
    """
    Pipeline original do TrackNet -> https://github.com/yastrebksv/TrackNet
    """
    print(f"Processando vídeo: {caminho_video}")

    #Carrega o vídeo
    frames, fps = read_video(caminho_video)

    #Roda a inferência
    ball_track, dists = infer_model(frames, modelo)

    #Remove frames com falha de posicionamento
    ball_track = remove_outliers(ball_track, dists)

    #interpola posições onde a bola não pode ser encontrada    
    subtracks = split_track(ball_track)
    for r in subtracks:
        ball_subtrack = ball_track[r[0]:r[1]]
        ball_subtrack = interpolation(ball_subtrack)
        ball_track[r[0]:r[1]] = ball_subtrack

    #salva um vídeo para visualização da inferência
    write_track(frames, ball_track, caminho_saida, fps)

    h_orig, w_orig = frames[0].shape[:2]
    resolucao_original = (w_orig, h_orig)
    
    return ball_track, resolucao_original

def reescalar_trajetoria(ball_track, resolucao_original, dim_modelo=(640, 360)):
    """
    Converte as coordenadas da inferência (640x360) de volta para a resolução original do vídeo
    """
    w_orig, h_orig = resolucao_original
    w_mod, h_mod = dim_modelo
    
    escala_x = w_orig / w_mod
    escala_y = h_orig / h_mod
    
    trajetoria_real = []
    for pt in ball_track:
        if pt[0] is not None and not torch.isnan(torch.tensor(pt[0])):
            x_real = pt[0] * escala_x
            y_real = pt[1] * escala_y
            trajetoria_real.append((x_real, y_real))
        else:
            trajetoria_real.append((None, None))
            
    return trajetoria_real

if __name__ == '__main__':
    video_esq = "data/test_videos/esq/"
    video_dir = "data/test/videos/dir/"
    pesos_modelo = "tracknet/model_best.pt" 
    
    # Configuração do Modelo (Carregado apenas 1x na GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Carregando modelo: {device.upper()}")
    
    modelo = BallTrackerNet()
    modelo.load_state_dict(torch.load(pesos_modelo, map_location=device))
    modelo = modelo.to(device)
    modelo.eval()

    # Processamento sequencial
    track_esq_bruto, res_esq = processar_camera(video_esq, "data/out_esq.avi", modelo)
    track_dir_bruto, res_dir = processar_camera(video_dir, "data/out_dir.avi", modelo)
    
    # Reescalar coordenadas para a fotogrametria (suas matrizes P1 e P2 esperam pontos em 4K/1080p)
    pontos_esq_2d = reescalar_trajetoria(track_esq_bruto, res_esq)
    pontos_dir_2d = reescalar_trajetoria(track_dir_bruto, res_dir)

    dados_exportacao = {
    "resolucao_esq": res_esq,
    "resolucao_dir": res_dir,
    "trajetoria_esq": pontos_esq_2d,
    "trajetoria_dir": pontos_dir_2d}

    caminho_json = "data/trajetorias_2d.json"
    with open(caminho_json, 'w') as f:
        json.dump(dados_exportacao, f, indent=4)

    print(f"Coordenadas 2D salvas com sucesso em '{caminho_json}'.")
    
    print("Processamento concluído. Verifique os vídeos gerados em 'data/'.")

