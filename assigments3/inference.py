import torch
from assigments3.train import CTCModel, inference_and_submit, N_MELS, VOCAB

def run_inference_only(model_path: str,
                       audio_dir: str = 'test',
                       output_csv: str = 'submission_2.csv'):
    # 1) Подготовить устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2) Воссоздать архитектуру и загрузить веса
    model = CTCModel(input_dim=N_MELS,
                     hidden_dim=256,
                     num_layers=3,
                     num_classes=len(VOCAB)).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 3) Запустить инференс и сохранить submission.csv
    inference_and_submit(model, audio_dir=audio_dir, device=device, output_csv=output_csv)

if __name__ == '__main__':
    # Укажите свой .pth файл
    run_inference_only(model_path='ctc_model2.pth',
                       audio_dir='.')
