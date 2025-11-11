import time

#Implementacja klasy gry
class Game:
    def __init__(self):
        self.Board = [' ' for _ in range(9)] #deklaracja i wypełnienie tablicy

    def print_board(self):
        print()
        for i in range(3): #3wiersze
            print(f"| {self.Board[i * 3]} | {self.Board[i * 3 + 1]} | {self.Board[i * 3 + 2]} |") #indeksy 1,2,3 w wierszu
        print()

    def available_moves(self):
        moves = []
        for i in range(9):
            if self.Board[i] == ' ':
                moves.append(i)
        return moves

    def make_move(self, index, player):
        if self.Board[index] == ' ':
            self.Board[index] = player
            return True
        return False

    def check_winner(self):
        win_patterns = [ #lista zwycięskich kombinacji
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # wiersze
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # kolumny
            [0, 4, 8], [2, 4, 6]              # przekątne
        ]

        for a, b, c in win_patterns: #indeksy pól
            if self.Board[a] != ' ' and self.Board[a] == self.Board[b] == self.Board[c]:
                return self.Board[a]

        return ' '

    def has_empty_squares(self): #bool
        return ' ' in self.Board

    def play(self, x_player, o_player):
        current_player = 'X'
        self.print_board()

        while self.has_empty_squares():
            move = x_player.get_move() if current_player == 'X' else o_player.get_move()
            self.make_move(move, current_player)
            self.print_board()

            winner = self.check_winner()
            if winner != ' ':
                print(f"Gracz {winner} wygrywa!")
                return

            current_player = 'O' if current_player == 'X' else 'X'
            time.sleep(0.7)

        print("Remis!")

#Implementacja klasy gracza
class Player:
    def __init__(self, letter, game):
        self.Letter = letter #symbol gracza
        self.GameRef = game #referencja do gry

    def get_move(self): #metoda do nadpisania w klasie dziedziczącej
        raise NotImplementedError

#Implementacja klasy gracza sterowanego przez algorytm minimax
class MiniMaxPlayer(Player):
    def __init__(self, letter, game, depth_limit): #konstruktor
        super().__init__(letter, game) #wywołanie konstruktora klasy bazowej
        self.depth_limit = depth_limit

    def get_move(self):
        best_move = -1
        best_score = float('-inf') if self.Letter == 'X' else float('inf') #maksymalizacja dla X

        for move in self.GameRef.available_moves():
            self.GameRef.make_move(move, self.Letter)
            score = self.minimax(self.depth_limit - 1, False)
            self.GameRef.Board[move] = ' '  #cofnięcie ruchu

            if self.Letter == 'X':
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move

    def minimax(self, depth, is_maximizing):
        winner = self.GameRef.check_winner()
        if winner == 'X':
            return 10
        if winner == 'O':
            return -10
        if not self.GameRef.has_empty_squares() or depth == 0:
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for move in self.GameRef.available_moves():
                self.GameRef.make_move(move, 'X')
                score = self.minimax(depth - 1, False)
                self.GameRef.Board[move] = ' '
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in self.GameRef.available_moves():
                self.GameRef.make_move(move, 'O')
                score = self.minimax(depth - 1, True)
                self.GameRef.Board[move] = ' '
                best_score = min(best_score, score)
            return best_score

#Implementacja głównej części programu
if __name__ == "__main__":
    game = Game()
    x_player = MiniMaxPlayer('X', game, depth_limit=10)
    o_player = MiniMaxPlayer('O', game, depth_limit=10)

    game.play(x_player, o_player)