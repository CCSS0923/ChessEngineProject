import torch
from models.chessnet import ChessNet


def main():
    device = torch.device("cpu")  # TS export는 cpu 기준으로 만들어도 됨
    model = ChessNet().to(device).eval()

    # 예시 입력 (dummy)
    example = torch.zeros(1, 18, 8, 8, device=device)

    # script 사용 (control-flow 문제 없음)
    scripted = torch.jit.script(model)

    out_path = "../checkpoints/chessnet_ts.pt"
    scripted.save(out_path)
    print("saved TorchScript model to:", out_path)


if __name__ == "__main__":
    main()
