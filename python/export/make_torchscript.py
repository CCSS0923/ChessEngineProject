import torch
from models.chessnet import ChessNet

def main():
    model = ChessNet().eval()
    example = torch.zeros(1,18,8,8)
    ts = torch.jit.script(model)
    ts.save("../../checkpoints/chessnet_ts.pt")
    print("saved chessnet_ts.pt")

if __name__ == "__main__":
    main()
