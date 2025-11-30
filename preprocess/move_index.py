import chess

MAX_MOVE_SPACE = 64 * 64  # 4096

def uci_to_index(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def index_to_uci(idx: int) -> chess.Move:
    from_sq = idx // 64
    to_sq = idx % 64
    return chess.Move(from_sq, to_sq)
